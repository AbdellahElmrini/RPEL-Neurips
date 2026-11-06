import torch
import numpy as np
from typing import List, Optional, Union, Any
from utils.array_ops import (
    compute_norm, argsort, cumsum, searchsorted, weighted_sum, weighted_mean, 
    clip_by_threshold, compute_rest_weight,
    compute_sorted_gradients_and_weights, find_weight_quantile_index
) 
from utils.misc import check_vectors_type
from utils.type_utils import convert_weights_to_type, ensure_scalar, convert_threshold_to_type, extract_device_and_dtype

def cs_plus(weights, gradients, byz_weights, **kwargs):

    distances = gradients.norm(dim=1)
    
    # Sort distances and rearrange weights accordingly
    sorted_indices = torch.argsort(distances)
    sorted_distances = distances[sorted_indices]
    sorted_weights = weights[sorted_indices]


    # Compute cumulative weights
    cumulative_weights = torch.cumsum(sorted_weights, dim=0)
    total_weight = cumulative_weights[-1]

    # Determine the quantile position
    target_weight = total_weight - 2*byz_weights 

    # Find the index where the cumulative weight exceeds the target weight
    idx = torch.searchsorted(cumulative_weights, target_weight) # nb: weights[idx:] >= byz_weights
    
    # Compute the clipping threshold
    if idx < 0:
        clipping_threshold = 0
    elif byz_weights == 0 or idx >= gradients.shape[0]:
        clipping_threshold = torch.inf

    elif cumulative_weights[idx] == target_weight:
        clipping_threshold = sorted_distances[idx]
    elif cumulative_weights[idx] > target_weight:
        if idx-1 >= 0:
            clipping_threshold = sorted_distances[idx-1]
        else:
            clipping_threshold =  0
    else:
        raise ValueError("Unexpected behavior in computing the adaptive clipping threshold")
    
    # Clip the gradients
    mask = distances[:, None].broadcast_to(gradients.shape) > clipping_threshold
    num_clipped = mask[:,0].sum().item()
    clipped_differences = torch.where(
        mask,  # Compare each norm to the threshold
        gradients * (clipping_threshold / distances[:, None]),  # Scale down the vector
        gradients  # Otherwise, keep it unchanged
    )

    return (weights[:, None] * clipped_differences).sum(dim=0), num_clipped

def cs_plus_bis(weights, gradients, byz_weights, **kwargs):

    # Adapted to also store the number of clipped gradients
    distances = gradients.norm(dim=1)
    
    # Sort distances and rearrange weights accordingly
    sorted_indices = torch.argsort(distances)
    sorted_distances = distances[sorted_indices]
    sorted_weights = weights[sorted_indices]


    # Compute cumulative weights
    cumulative_weights = torch.cumsum(sorted_weights, dim=0)
    total_weight = cumulative_weights[-1]

    # Determine the quantile position
    target_weight = total_weight - 2*byz_weights 

    # Find the index where the cumulative weight exceeds the target weight
    idx = torch.searchsorted(cumulative_weights, target_weight) # nb: weights[idx:] >= byz_weights
    
    # Compute the clipping threshold
    if idx < 0:
        clipping_threshold = 0
    elif byz_weights == 0 or idx >= gradients.shape[0]:
        clipping_threshold = torch.inf

    elif cumulative_weights[idx] == target_weight:
        clipping_threshold = sorted_distances[idx]
    elif cumulative_weights[idx] > target_weight:
        if idx-1 >= 0:
            clipping_threshold = sorted_distances[idx-1]
        else:
            clipping_threshold =  0
    else:
        raise ValueError("Unexpected behavior in computing the adaptive clipping threshold")
    
    # Clip the gradients
    mask = distances[:, None].broadcast_to(gradients.shape) > clipping_threshold
    num_clipped = mask[:,0].sum()
    #print("clipping threshold:", clipping_threshold)
    #print("num clipped:", mask.sum().item())
    #print("mask:", mask)
    clipped_differences = torch.where(
        mask,  # Compare each norm to the threshold
        gradients * (clipping_threshold / distances[:, None]),  # Scale down the vector
        gradients  # Otherwise, keep it unchanged
    )

    return (weights[:, None] * clipped_differences).sum(dim=0), num_clipped

    
### GTS, leads to NNA

def gts(weights, gradients, byz_weights, **kwargs):

    distances = gradients.norm(dim=1)
    
    # Sort distances and rearrange weights accordingly
    sorted_indices = torch.argsort(distances)
    sorted_weights = weights[sorted_indices]

    # Compute cumulative weights
    cumulative_weights = torch.cumsum(sorted_weights, dim=0)
    total_weight = cumulative_weights[-1]

    # Determine the quantile position
    target_weight = total_weight - byz_weights 

    # Find the index where the cumulative weight exceeds the target weight
    idx = torch.searchsorted(cumulative_weights, target_weight) # weights(idx:) >= byz_weights
    
    rest_weight = 0
    if cumulative_weights[idx] > target_weight:
        rest_weight =  cumulative_weights[idx] - target_weight
    
    sorted_gradients = gradients[sorted_indices,:]
    return (sorted_weights[:idx, None] * sorted_gradients[:idx,:]).sum(dim=0) + sorted_gradients[idx,:] * rest_weight, 0


### CShe practical from [He, Karimirredy and Jaggi 2022]

def cs_he(weights, gradients, byz_weights, **kwargs):

    distances = gradients.norm(dim=1)
    
    # Sort distances and rearrange weights accordingly
    sorted_indices = torch.argsort(distances)
    sorted_distances = distances[sorted_indices]
    sorted_weights = weights[sorted_indices]

    # Compute cumulative weights
    cumulative_weights = torch.cumsum(sorted_weights, dim=0)
    total_weight = cumulative_weights[-1]

    # Determine the quantile position
    target_weight = total_weight - byz_weights 

    # Find the index where the cumulative weight exceeds the target weight
    idx = torch.searchsorted(cumulative_weights, target_weight) # weights(idx:) >= byz_weights
    
    # Compute the adaptive clipping threshold of He et al. 
    if idx == 0:
        clipping_threshold = 0
    elif byz_weights == 0 or idx >= gradients.shape[0]:
        clipping_threshold = torch.inf
    else:
        clipping_threshold = ((sorted_weights[:idx] * sorted_distances[:idx]**2).sum(dim=0) / byz_weights).sqrt().item()

    # Clip the gradients
    mask = distances[:, None].broadcast_to(gradients.shape) > clipping_threshold
    num_clipped = mask[:,0].sum().item()
    clipped_gradients = torch.where(
        mask,  # Compare each norm to the threshold
        gradients * (clipping_threshold / distances[:, None]),  # Scale down the vector
        gradients  # Otherwise, keep it unchanged
    )
    

    return (weights[:, None] * clipped_gradients).sum(dim=0), num_clipped




class RTC(object):
    """
    RTC (Remove-then-Clip) robust aggregator - Pure implementation.
    
    This implements the original Remove-then-Clip aggregator algorithm,
    which is NOT part of the RG framework. It directly aggregates gradients
    using a two-phase approach:
    1. Remove phase: Iteratively removes neighbors with largest distance from local model
       until total removed weight >= byzantine_weight_budget
    2. Clip phase: Computes adaptive clipping radius from remaining neighbors
    3. Aggregate: Clips remaining neighbors and replaces removed with local model

    
    Parameters
    ----------
    None 
    
    Methods
    -------
    __call__(gradients, weights, honest_index, byzantine_weight_budget, communication_step=1.0)
        Perform RTC aggregation directly on gradients.
    """
    
    def __init__(self):
        """Initialize RTC aggregator."""
        pass
    
    @staticmethod
    def _rtc_remove_phase(
        gradients_matrix: Union[torch.Tensor, np.ndarray],
        local_gradient: Union[torch.Tensor, np.ndarray],
        weights: Union[torch.Tensor, np.ndarray],
        delta_max: float,
        tools: Any
    ) -> tuple:
        """
        RTC Remove phase: Iteratively remove neighbors until removed weight >= delta_max.
        
        Parameters
        ----------
        gradients_matrix : torch.Tensor or numpy.ndarray
            Stacked gradients, shape (n, d)
        local_gradient : torch.Tensor or numpy.ndarray
            Local node's gradient, shape (d,)
        weights : torch.Tensor or numpy.ndarray
            Weights for each gradient, shape (n,)
        delta_max : float
            Maximum Byzantine weight budget
        tools : module
            Either numpy or torch_tools module
        
        Returns
        -------
        tuple
            (remaining, differences, removed_weight)
            remaining: boolean mask of remaining neighbors
            differences: differences from local gradient, shape (n, d)
            removed_weight: total weight removed
        """
        n = gradients_matrix.shape[0]
        
        # Compute distances from local gradient
        if tools == np:
            local_grad_expanded = local_gradient[np.newaxis, :]
            differences = gradients_matrix - local_grad_expanded
            removed = np.zeros(n, dtype=bool)
            candidates = np.arange(n)
        else:
            differences = gradients_matrix - local_gradient.unsqueeze(0)
            # gradients_matrix is guaranteed to be torch.Tensor when tools != np
            device, _ = extract_device_and_dtype(gradients_matrix)
            if device is None:
                device = torch.device('cpu')
            removed = torch.zeros(n, dtype=torch.bool, device=device)
            candidates = torch.arange(n, device=device)
        
        distances = compute_norm(differences, tools, axis=1)
        removed_weight = 0.0
        
        while True:
            if removed_weight >= delta_max or len(candidates) == 0:
                break
            
            candidate_distances = distances[candidates]
            if len(candidate_distances) == 0:
                break
            
            if tools == np:
                farthest_idx_in_candidates = np.argmax(candidate_distances)
                farthest = candidates[farthest_idx_in_candidates]
            else:
                farthest_idx_in_candidates = torch.argmax(candidate_distances)
                farthest = candidates[farthest_idx_in_candidates].item()
            
            w = ensure_scalar(weights[farthest])
            
            if removed_weight + w > delta_max:
                break
            
            removed[farthest] = True
            removed_weight += w
            if tools == np:
                candidates = candidates[candidates != farthest]
            else:
                candidates = candidates[candidates != farthest]
        
        remaining = ~removed
        return remaining, differences, removed_weight
    
    @staticmethod
    def _rtc_compute_adaptive_radius(
        remaining_weights: Union[torch.Tensor, np.ndarray],
        remaining_differences: Union[torch.Tensor, np.ndarray],
        delta_max: float,
        tools: Any,
        reference_tensor: Optional[Union[torch.Tensor, np.ndarray]] = None
    ) -> Union[float, torch.Tensor, np.ndarray]:
        """
        Compute adaptive clipping radius tau_i from remaining neighbors.
        
        Parameters
        ----------
        remaining_weights : torch.Tensor or numpy.ndarray
            Weights of remaining neighbors, shape (r,)
        remaining_differences : torch.Tensor or numpy.ndarray
            Differences of remaining neighbors, shape (r, d)
        delta_max : float
            Maximum Byzantine weight budget
        tools : module
            Either numpy or torch_tools module
        reference_tensor : torch.Tensor or numpy.ndarray, optional
            Reference tensor for device/dtype
        
        Returns
        -------
        float, torch.Tensor, or numpy.ndarray
            Clipping radius tau_i
        """
        remaining_distances = compute_norm(remaining_differences, tools, axis=1)
        remaining_distances_sq = remaining_distances ** 2
        
        if tools == np:
            sum_weighted_sqdist = np.sum(remaining_weights * remaining_distances_sq)
            tau_i = np.sqrt(sum_weighted_sqdist / delta_max)
        else:
            sum_weighted_sqdist = (remaining_weights * remaining_distances_sq).sum()
            tau_i = torch.sqrt(sum_weighted_sqdist / delta_max)
        
        return tau_i
    
    @staticmethod
    def _rtc_clip_phase(
        remaining_differences: Union[torch.Tensor, np.ndarray],
        remaining_weights: Union[torch.Tensor, np.ndarray],
        tau_i: Union[float, torch.Tensor, np.ndarray],
        local_gradient: Union[torch.Tensor, np.ndarray],
        tools: Any
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        RTC Clip phase: Clip remaining differences and aggregate.
        
        Parameters
        ----------
        remaining_differences : torch.Tensor or numpy.ndarray
            Differences of remaining neighbors, shape (r, d)
        remaining_weights : torch.Tensor or numpy.ndarray
            Weights of remaining neighbors, shape (r,)
        tau_i : float, torch.Tensor, or numpy.ndarray
            Clipping radius
        local_gradient : torch.Tensor or numpy.ndarray
            Local gradient for shape reference
        tools : module
            Either numpy or torch_tools module
        
        Returns
        -------
        torch.Tensor or numpy.ndarray
            Aggregated update
        """
        rem_distances = compute_norm(remaining_differences, tools, axis=1)
        
        if tools == np:
            if tau_i > 0:
                norms = rem_distances[:, np.newaxis]
                factors = np.clip(tau_i / norms, a_min=None, a_max=1.0)
                factors[np.isnan(factors)] = 0.0
                clipped = remaining_differences * factors
            else:
                clipped = np.zeros_like(remaining_differences)
        else:
            if tau_i > 0:
                norms = rem_distances.unsqueeze(1)
                factors = torch.clamp(tau_i / norms, max=1.0)
                factors[torch.isnan(factors)] = 0.0
                clipped = remaining_differences * factors
            else:
                clipped = torch.zeros_like(remaining_differences)
        
        return weighted_sum(remaining_weights, clipped, tools, axis=0)
    
    def __call__(
        self,
        gradients: List[Union[torch.Tensor, np.ndarray]],
        local_gradient,
        weights: List,
        byzantine_weight_budget: float,
        communication_step: float = 1.0
    ) :#-> Union[torch.Tensor, np.ndarray]:
        """
        Perform RTC aggregation directly on gradients.
        
        This method implements the original RTC algorithm without the RG framework.
        It takes gradients directly and returns the aggregated gradient.
        
        Parameters
        ----------
        gradients : list of torch.Tensor or numpy.ndarray
            List of gradient vectors from neighbors, including self.
            Shape: each gradient is (param_dim,)
            Can contain either torch.Tensor or numpy.ndarray, but all must be same type.
        weights : list, torch.Tensor, or numpy.ndarray
            Mixing matrix weights for each gradient in gradients.
            Must have same length as gradients.
        honest_index : int
            Index of the local node's gradient in the gradients list.
        byzantine_weight_budget : float
            Upper bound on total Byzantine weight (delta_max in the paper).
            Neighbors will be removed until removed weight >= byzantine_weight_budget.
        communication_step : float, optional
            Communication step size. Default is 1.0.
            The final result is: local_gradient + communication_step * aggregated_update
        
        Returns
        -------
        torch.Tensor or numpy.ndarray
            Aggregated gradient vector using RTC algorithm (same type as input).
            Final update: local_gradient + communication_step * aggregated_update
        """
        # Validate inputs
        if not isinstance(gradients, list) or len(gradients) == 0:
            raise ValueError("gradients must be a non-empty list")
        
        if len(weights) != len(gradients):
            raise ValueError(f"Weights length ({len(weights)}) must match gradients length ({len(gradients)})")
        
        #if not isinstance(honest_index, int) or honest_index < 0 or honest_index >= len(gradients):
        #    raise ValueError(f"Invalid honest_index: {honest_index}, must be 0 ≤ honest_index < {len(gradients)}")
        
        # Use check_vectors_type to handle both numpy and torch (detects type and returns tools)
        tools, gradients_matrix = check_vectors_type(gradients)
        
        # Extract local gradient from gradients matrix
        #local_gradient = gradients_matrix[honest_index, :]
        
        # Convert weights to appropriate type using utility function
        weights = convert_weights_to_type(weights, tools, reference_tensor=gradients_matrix)
        
        n = gradients_matrix.shape[0]
        
        # Convert byzantine_weight_budget to float
        delta_max = float(byzantine_weight_budget)
        
        # If no Byzantine weight budget (delta_max <= 0): no removal and no clipping
        # Return simple weighted average
        if delta_max <= 0:
            total_weight = weights.sum() if tools != np else np.sum(weights)
            if total_weight > 0:
                weighted_mean_result = weighted_mean(weights, gradients_matrix, tools, axis=0)
                diff = tools.subtract(weighted_mean_result, local_gradient)
                return tools.add(local_gradient, tools.multiply(diff, communication_step))
            else:
                return local_gradient
        
        # Phase 1: Remove phase
        remaining, differences, _ = self._rtc_remove_phase(
            gradients_matrix, local_gradient, weights, delta_max, tools
        )
        self.num_removed = (~remaining).sum().item() if tools != np else np.sum(~remaining)
        
        # Phase 2: Compute adaptive clipping radius
        if tools == np:
            remaining_sum = np.sum(remaining)
        else:
            remaining_sum = remaining.sum()
        
        if remaining_sum > 0:
            remaining_weights = weights[remaining]
            remaining_differences = differences[remaining, :]
            tau_i = self._rtc_compute_adaptive_radius(
                remaining_weights, remaining_differences, delta_max, tools, reference_tensor=gradients_matrix
            )
        else:
            if tools == np:
                tau_i = np.float32(0.0)
            else:
                # gradients_matrix is guaranteed to be torch.Tensor when tools != np
                device, dtype = extract_device_and_dtype(gradients_matrix)
                if device is None:
                    device = torch.device('cpu')
                if dtype is None:
                    dtype = torch.float32
                tau_i = torch.tensor(0.0, device=device, dtype=dtype)
        
        # Phase 3: Clip and aggregate
        if remaining_sum > 0:
            remaining_weights = weights[remaining]
            remaining_differences = differences[remaining, :]
            aggregated_update = self._rtc_clip_phase(
                remaining_differences, remaining_weights, tau_i, local_gradient, tools
            )
        else:
            if tools == np:
                aggregated_update = np.zeros_like(local_gradient)
            else:
                aggregated_update = torch.zeros_like(local_gradient)
        
        # Final result: x_i + communication_step * aggregated_update
        scaled_update = tools.multiply(aggregated_update, communication_step)
        return tools.add(local_gradient, scaled_update)