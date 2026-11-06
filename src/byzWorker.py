from networkx import neighbors
from .byz_attacks import ByzantineAttack
from .robust_aggregators import RobustAggregator
import torch
from typing import Dict, Union, Tuple, Callable

from src.robust_summations import cs_plus, gts, cs_he, RTC
from utils.attack_utils import line_search_optimize


methods_dict={"cs+":cs_plus, "cs_he":cs_he, "gts":gts}

class ByzantineWorker(object):
    """A Byzantine worker for distributed training."""

    def __init__(self, nb_workers, nb_decl_byz, nb_real_byz, attack, aggregator, second_aggregator, server_clip, bucket_size,
                 model_size, mimic_learning_phase, gradient_clip, device):

        #JS: Instantiate the robust aggregator to be used (in particular to be used for auto ALIE and auto FOE)
        robust_aggregator = RobustAggregator(aggregator, second_aggregator, server_clip, nb_workers, nb_decl_byz, bucket_size, model_size, device)
        #JS: Instantiate the Byzantine attack to be used
        self.byzantine_attack = ByzantineAttack(attack, nb_real_byz, model_size, device, mimic_learning_phase, gradient_clip, robust_aggregator)

    def compute_byzantine_vectors(self, honest_vectors, grads_flipped, current_step):
        return self.byzantine_attack.generate_byzantine_vectors(honest_vectors, grads_flipped, current_step)

    def compute_byzantine_running_mean_and_var(self, honest_running_mean, honest_running_var, current_step):
        byzantine_dict_mean = dict()
        for batch_layer in honest_running_mean[0].keys():
            running_means = [honest_running_mean[worker][batch_layer] for worker in range(len(honest_running_mean))]
            byzantine_dict_mean[batch_layer] = self.byzantine_attack.generate_byzantine_vectors(running_means, None, current_step)

        byzantine_dict_var = dict()
        for batch_layer in honest_running_var[0].keys():
            running_vars = [honest_running_var[worker][batch_layer] for worker in range(len(honest_running_var))]
            byzantine_dict_var[batch_layer] =  self.byzantine_attack.generate_byzantine_vectors(running_vars, None, current_step)

        return byzantine_dict_mean, byzantine_dict_var
    

class DecByzantineWorker(object):
    def __init__(self, target, honest_neighbors, nb_honest, nb_byz_neighbors, pivot_params, network, device, epsilon = 1):
        self.target = target
        self.nb_honest = nb_honest  
        self.nb_byz_neighbors = nb_byz_neighbors
        self.pivot_params = pivot_params
        self.network = network
        self.device = device 
        self.epsilon = epsilon
        self.honest_neighbors=  honest_neighbors

        
    def compute_byzantine_vectors(self, honest_local_params):
        W_i = self.network.weights(self.target)
        
        total_byz_weights = W_i[[k for k in self.network.neighbors(self.target) if k >= self.nb_honest]].sum()
        honest_local_params = torch.stack(honest_local_params)
        differences = honest_local_params - self.pivot_params
        #self.comm_graph.weights(self.worker_id)[self.neighbors] @ differences 
        #byzantine_vector = self.pivot_params - (W_i[self.neighbors]/ total_byz_weights)[:, None] @ differences
        byzantine_vector = self.pivot_params - self.epsilon/ total_byz_weights* torch.matmul((W_i[self.honest_neighbors] ).unsqueeze(0), differences)
        #byzantine_vector = - self.epsilon/ total_byz_weights* torch.matmul((W_i[self.honest_neighbors] ).unsqueeze(0), differences)

        byzantine_vector = byzantine_vector.squeeze(0)
        return [byzantine_vector] * self.nb_byz_neighbors
    


class OptimalDissensus(object):
    
    """
    True optimal dissensus attack with epsilon tuning.

    - Uses only neighbors of the attacked node
    - Byzantine update is epsilon * dissensus_direction
    - Dissensus direction is fixed
    - Epsilon is optimized using the actual aggregator
    """

    def __init__(
        self,
        target,
        honest_neighbors,
        nb_honest,
        nb_byz_neighbors,
        pivot_params,
        network,
        method,
        evals: int = 20,
        start: float = 0.0,
        delta: float = 1.0,
        ratio: float = 0.8,
    ):
        self.honest_neighbors = honest_neighbors
        self.target = target
        self.nb_honest = nb_honest
        self.nb_byz_neighbors = nb_byz_neighbors
        self.pivot_params = pivot_params
        self.network = network
        self.method=method
        
        self.evals = evals
        self.start = start
        self.delta = delta
        self.ratio = ratio

    # ------------------------------------------------------------------
    # 1. Dissensus direction (computed ONCE)
    # ------------------------------------------------------------------
    def _dissensus_direction(
        self,
        honest_local_params, 
    ) -> torch.Tensor:

        #honest_local_params = torch.stack(honest_local_params)
        #W_i = network.get_mixing_row_weights(attacked_node)
        W_i = self.network.weights(self.target)
        
        


        if self.nb_byz_neighbors == 0:
            return torch.zeros_like(self.pivot_params)
        b_i = W_i[[k for k in self.network.neighbors(self.target) if k >= self.nb_honest]].sum()
        if b_i == 0:
            return torch.zeros_like(self.pivot_params)

        #honest_local_params = torch.stack(honest_local_params)
        differences = honest_local_params - self.pivot_params

        numerator =  torch.matmul((W_i[self.honest_neighbors] ).unsqueeze(0), differences)



        return - numerator / b_i

    # ------------------------------------------------------------------
    # 2. Evaluate a given epsilon using the real aggregator
    # ------------------------------------------------------------------
    def _evaluate_eps(
        self,
        epsilon: float,
        direction: torch.Tensor,
        honest_params,
        honest_local_params,
    ) -> float:

        W_i = self.network.weights(self.target)


        #Computing the honest target aggregation
        #honest_local_params = torch.stack(honest_local_params)
        differences = honest_local_params - self.pivot_params
        honest_weights = W_i[self.honest_neighbors]
        honest_target = torch.matmul(honest_weights.unsqueeze(0), differences).squeeze(0)




        
        neighbors = list(self.network.neighbors(self.target))
        neighbors.append(self.target)

        # Honest aggregation target
        
        
        


        # Byzantine gradients: epsilon * direction
        byz_grad = epsilon * direction
        
        b_i = W_i[[k for k in self.network.neighbors(self.target) if k >= self.nb_honest]].sum()
        #honest_eighwts = W_i[self.honest_neighbors]



        # Build mapping of neighbor index to byzantine param index
        byz_idx = 0
        worker_params = []
        for neighbor in neighbors:
            if neighbor < self.nb_honest:
                worker_params.append(honest_params[neighbor])
            else:
                worker_params.append(byz_grad.squeeze(0))
                byz_idx += 1


        worker_params = torch.stack(worker_params)

        
        corrupted_differences = worker_params - self.pivot_params
        
 


        attacked_result, num_clipped = methods_dict[self.method](weights=self.network.weights(self.target)[neighbors].clone().detach().requires_grad_(False),#.to(self.device), 
                                        gradients=corrupted_differences, 
                                        byz_weights=b_i) 
        

        diff = attacked_result - honest_target
        return float(torch.dot(diff, diff))

    # ------------------------------------------------------------------
    # 3. Main call
    # ------------------------------------------------------------------
    def __call__(
        self,
        honest_params,
    ) -> torch.Tensor:
        
        neighbors = list(self.network.neighbors(self.target))
        neighbors.append(self.target)
        honest_local_params = [honest_params[i] for i in neighbors if i < self.nb_honest]
        honest_local_params = torch.stack(honest_local_params)
        # Compute dissensus direction ONCE
        direction = self._dissensus_direction(
            honest_local_params,
        )

        if torch.norm(direction) == 0:
            return direction

        # Line search on epsilon
        eval_fn = lambda eps: self._evaluate_eps(
            eps,
            direction,
            honest_params,
            honest_local_params,
        )

        best_eps = line_search_optimize(
            eval_fn,
            start=self.start,
            delta=self.delta,
            ratio=self.ratio,
            evals=self.evals,
        )
        print("Best eps :" + str(best_eps))
        byz_vector = best_eps * direction
        return  [byz_vector.squeeze(0)] * self.nb_byz_neighbors 
