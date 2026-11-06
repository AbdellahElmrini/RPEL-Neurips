"""
Array Operations Utilities for ByzDL

This module provides unified helper functions for common numpy/torch operations,
abstracting away the differences between numpy and torch APIs.
"""

import torch
import numpy as np
from typing import Union, Any, Optional


def compute_norm(
    vectors: Union[torch.Tensor, np.ndarray],
    tools: Any,
    axis: int = 1
) -> Union[torch.Tensor, np.ndarray]:
    """
    Compute L2 norm along specified axis.
    
    Parameters
    ----------
    vectors : torch.Tensor or numpy.ndarray
        Vectors to compute norm for
    tools : module
        Either numpy or torch_tools module
    axis : int, optional
        Axis along which to compute norm. Default 1.
    
    Returns
    -------
    torch.Tensor or numpy.ndarray
        Norm values along specified axis
    """
    if tools == np:
        return np.linalg.norm(vectors, axis=axis)
    else:
        return vectors.norm(dim=axis)


def argsort(
    values: Union[torch.Tensor, np.ndarray],
    tools: Any
) -> Union[torch.Tensor, np.ndarray]:
    """
    Return indices that would sort the values.
    
    Parameters
    ----------
    values : torch.Tensor or numpy.ndarray
        Values to sort
    tools : module
        Either numpy or torch_tools module
    
    Returns
    -------
    torch.Tensor or numpy.ndarray
        Indices that sort the values
    """
    if tools == np:
        return np.argsort(values)
    else:
        return torch.argsort(values)


def cumsum(
    values: Union[torch.Tensor, np.ndarray],
    tools: Any,
    axis: int = 0
) -> Union[torch.Tensor, np.ndarray]:
    """
    Compute cumulative sum along specified axis.
    
    Parameters
    ----------
    values : torch.Tensor or numpy.ndarray
        Values to compute cumulative sum for
    tools : module
        Either numpy or torch_tools module
    axis : int, optional
        Axis along which to compute cumulative sum. Default 0.
    
    Returns
    -------
    torch.Tensor or numpy.ndarray
        Cumulative sum along specified axis
    """
    if tools == np:
        return np.cumsum(values, axis=axis)
    else:
        return torch.cumsum(values, dim=axis)


def searchsorted(
    sorted_array: Union[torch.Tensor, np.ndarray],
    value: Union[float, int, torch.Tensor, np.ndarray],
    tools: Any
) -> Union[int, torch.Tensor]:
    """
    Find indices where values should be inserted to maintain order.
    
    Parameters
    ----------
    sorted_array : torch.Tensor or numpy.ndarray
        Sorted array to search in
    value : float, int, torch.Tensor, or numpy.ndarray
        Value(s) to insert
    tools : module
        Either numpy or torch_tools module
    
    Returns
    -------
    int or torch.Tensor
        Index or indices where value should be inserted.
        For torch, returns int (converted from tensor if needed).
    """
    if tools == np:
        return np.searchsorted(sorted_array, value)
    else:
        idx = torch.searchsorted(sorted_array, value)
        # Convert to int if it's a tensor
        if isinstance(idx, torch.Tensor):
            return idx.item() if idx.numel() == 1 else int(idx)
        return idx


def weighted_sum(
    weights: Union[torch.Tensor, np.ndarray],
    values: Union[torch.Tensor, np.ndarray],
    tools: Any,
    axis: int = 0
) -> Union[torch.Tensor, np.ndarray]:
    """
    Compute weighted sum along specified axis.
    
    Parameters
    ----------
    weights : torch.Tensor or numpy.ndarray
        Weights for each value, shape (n,)
    values : torch.Tensor or numpy.ndarray
        Values to weight, shape (n, d) or (n,)
    tools : module
        Either numpy or torch_tools module
    axis : int, optional
        Axis along which to sum. Default 0.
    
    Returns
    -------
    torch.Tensor or numpy.ndarray
        Weighted sum along specified axis
    """
    if tools == np:
        # Expand weights for broadcasting: (n,) -> (n, 1)
        weights_expanded = weights[:, np.newaxis] if values.ndim > 1 else weights
        return np.sum(weights_expanded * values, axis=axis)
    else:
        # Expand weights for broadcasting: (n,) -> (n, 1)
        weights_expanded = weights[:, None] if values.dim() > 1 else weights
        return (weights_expanded * values).sum(dim=axis)


def weighted_mean(
    weights: Union[torch.Tensor, np.ndarray],
    values: Union[torch.Tensor, np.ndarray],
    tools: Any,
    axis: int = 0,
    epsilon: float = 0.0
) -> Union[torch.Tensor, np.ndarray]:
    """
    Compute weighted mean (normalized by sum of weights).
    
    Equivalent to weighted_average: sum(weights * values) / (sum(weights) + epsilon)
    
    Parameters
    ----------
    weights : torch.Tensor or numpy.ndarray
        Weights for each value, shape (n,)
    values : torch.Tensor or numpy.ndarray
        Values to weight, shape (n, d) or (n,)
    tools : module
        Either numpy or torch_tools module
    axis : int, optional
        Axis along which to compute mean. Default 0.
    epsilon : float, optional
        Small value added to denominator for numerical stability. Default 0.0.
    
    Returns
    -------
    torch.Tensor or numpy.ndarray
        Weighted mean along specified axis
    """
    weighted_sum_result = weighted_sum(weights, values, tools, axis)
    weights_sum = tools.sum(weights)
    return tools.divide(weighted_sum_result, weights_sum + epsilon)


def weighted_variance(
    weights: Union[torch.Tensor, np.ndarray],
    values: Union[torch.Tensor, np.ndarray],
    tools: Any,
    axis: int = 0
) -> Union[torch.Tensor, np.ndarray]:
    """
    Compute weighted variance along specified axis.
    
    Formula: sum(weights * (values - weighted_mean)^2) / sum(weights)
    where weighted_mean = sum(weights * values) / sum(weights)
    
    Parameters
    ----------
    weights : torch.Tensor or numpy.ndarray
        Weights for each value, shape (n,)
    values : torch.Tensor or numpy.ndarray
        Values to compute variance for, shape (n, d) or (n,)
    tools : module
        Either numpy or torch_tools module
    axis : int, optional
        Axis along which to compute variance. Default 0.
    
    Returns
    -------
    torch.Tensor or numpy.ndarray
        Weighted variance along specified axis
    """
    if tools == np:
        # Weighted mean: sum(weights * values) / sum(weights)
        weights_sum = np.sum(weights)
        if values.ndim > 1:
            weighted_mean = np.sum(weights[:, np.newaxis] * values, axis=axis) / weights_sum
        else:
            weighted_mean = np.sum(weights * values, axis=axis) / weights_sum
        # Weighted variance: sum(weights * (values - mean)^2) / sum(weights)
        diff = values - weighted_mean
        if values.ndim > 1:
            var = np.sum(weights[:, np.newaxis] * diff ** 2, axis=axis) / weights_sum
        else:
            var = np.sum(weights * diff ** 2, axis=axis) / weights_sum
    else:
        # Weighted mean: sum(weights * values) / sum(weights)
        weights_sum = weights.sum()
        if values.dim() > 1:
            weighted_mean = (weights[:, None] * values).sum(dim=axis) / weights_sum
        else:
            weighted_mean = (weights * values).sum(dim=axis) / weights_sum
        # Weighted variance: sum(weights * (values - mean)^2) / sum(weights)
        diff = values - weighted_mean
        if values.dim() > 1:
            var = ((weights[:, None] * diff ** 2).sum(dim=axis)) / weights_sum
        else:
            var = ((weights * diff ** 2).sum(dim=axis)) / weights_sum
    
    return var


def clip_by_threshold(
    values: Union[torch.Tensor, np.ndarray],
    distances: Union[torch.Tensor, np.ndarray],
    threshold: Union[float, torch.Tensor, np.ndarray],
    tools: Any
) -> Union[torch.Tensor, np.ndarray]:
    """
    Clip values by threshold based on distances.
    
    For values where distance > threshold, scale them by threshold/distance.
    Otherwise, keep original values.
    
    Parameters
    ----------
    values : torch.Tensor or numpy.ndarray
        Values to clip, shape (n, d)
    distances : torch.Tensor or numpy.ndarray
        Distances for each value, shape (n,)
    threshold : float, torch.Tensor, or numpy.ndarray
        Clipping threshold
    tools : module
        Either numpy or torch_tools module
    
    Returns
    -------
    torch.Tensor or numpy.ndarray
        Clipped values
    """
    if tools == np:
        distances_expanded = distances[:, np.newaxis]
        mask = distances_expanded > threshold
        return np.where(
            mask,
            values * (threshold / distances_expanded),
            values
        )
    else:
        mask = distances[:, None].broadcast_to(values.shape) > threshold
        return torch.where(
            mask,
            values * (threshold / distances[:, None]),
            values
        )


def compute_rest_weight(
    cumulative_weights: Union[torch.Tensor, np.ndarray],
    target_weight: float,
    idx: int,
    tools: Any,
    reference_tensor: Optional[Union[torch.Tensor, np.ndarray]] = None
) -> Union[float, torch.Tensor]:
    """
    Compute the partial weight when cumulative weight exceeds target at index.
    
    This is used in aggregators like GTSRG when the cumulative weight at index idx
    exceeds the target weight, requiring a partial weight computation.
    
    Parameters
    ----------
    cumulative_weights : torch.Tensor or numpy.ndarray
        Cumulative weights array, shape (n,)
    target_weight : float
        Target weight threshold
    idx : int
        Index where cumulative weight exceeds target
    tools : module
        Either numpy or torch_tools module
    reference_tensor : torch.Tensor or numpy.ndarray, optional
        Reference tensor/array to extract device and dtype from.
        If None and target is torch, uses CPU and float32.
    
    Returns
    -------
    float or torch.Tensor
        Rest weight (partial weight). Returns 0.0 if idx doesn't exceed target.
    """
    if tools == np:
        rest_weight = np.float32(0.0)
        if idx < len(cumulative_weights) and cumulative_weights[idx] > target_weight:
            rest_weight = cumulative_weights[idx] - target_weight
        return rest_weight
    else:
        # For torch, create tensor with appropriate device/dtype
        if reference_tensor is not None and isinstance(reference_tensor, torch.Tensor):
            device = reference_tensor.device
            dtype = reference_tensor.dtype
        else:
            device = torch.device('cpu')
            dtype = torch.float32
        
        rest_weight = torch.tensor(0.0, device=device, dtype=dtype)
        if idx < len(cumulative_weights) and cumulative_weights[idx] > target_weight:
            rest_weight = cumulative_weights[idx] - target_weight
        return rest_weight


def compute_sorted_gradients_and_weights(
    gradients: Union[torch.Tensor, np.ndarray],
    weights: Union[torch.Tensor, np.ndarray],
    distances: Union[torch.Tensor, np.ndarray],
    tools: Any
) -> tuple:
    """
    Sort gradients and weights by distances.
    
    Common pattern used in aggregators (CSplusRG, GTSRG, CSheRG) to sort
    gradients and weights based on distance from a reference point.
    
    Parameters
    ----------
    gradients : torch.Tensor or numpy.ndarray
        Gradients to sort, shape (n, d)
    weights : torch.Tensor or numpy.ndarray
        Weights to sort, shape (n,)
    distances : torch.Tensor or numpy.ndarray
        Distances for sorting, shape (n,)
    tools : module
        Either numpy or torch_tools module
    
    Returns
    -------
    tuple
        (sorted_indices, sorted_gradients, sorted_weights, sorted_distances)
        All sorted by distances in ascending order
    """
    sorted_indices = argsort(distances, tools)
    sorted_gradients = gradients[sorted_indices, :]
    sorted_weights = weights[sorted_indices]
    sorted_distances = distances[sorted_indices]
    return sorted_indices, sorted_gradients, sorted_weights, sorted_distances


def find_weight_quantile_index(
    cumulative_weights: Union[torch.Tensor, np.ndarray],
    target_weight: float,
    tools: Any
) -> int:
    """
    Find the index where cumulative weight exceeds target weight.
    
    Common pattern used in aggregators to find quantile positions based on
    cumulative weights.
    
    Parameters
    ----------
    cumulative_weights : torch.Tensor or numpy.ndarray
        Cumulative weights array, shape (n,)
    target_weight : float
        Target weight threshold
    tools : module
        Either numpy or torch_tools module
    
    Returns
    -------
    int
        Index where cumulative weight exceeds target weight
    """
    return searchsorted(cumulative_weights, target_weight, tools)

