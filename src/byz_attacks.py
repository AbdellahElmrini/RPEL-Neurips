import torch, copy

from . import misc

def signflipping(honest_vectors, **kwargs):
    return torch.stack(honest_vectors).mean(dim=0).mul(-1)


def labelflipping(flipped_vectors, **kwargs):
    return torch.stack(flipped_vectors).mean(dim=0)


def fall_of_empires(honest_vectors, attack_factor=3, negative=False, **kwargs):
    #JS: negative controls the sign of the attack_factor
    #JS: attack_factor controls the magnitude of the attack_vector to be added to avg_honest_vector
    if negative:
        attack_factor = - attack_factor
    return torch.stack(honest_vectors).mean(dim=0).mul(1 - attack_factor)


def auto_FOE(honest_vectors, aggregator, nb_byz, gradient_clip, **kwargs):
    avg_honest_vector = torch.stack(honest_vectors).mean(dim=0)
    def eval_factor_FOE(factor):
        temp_vectors = [vector.clone() for vector in honest_vectors]
        #temp_vectors = copy.deepcopy(honest_vectors)
        byzantine_vector = fall_of_empires(temp_vectors, attack_factor=factor)
        if gradient_clip is not None:
            byzantine_vector = misc.clip_vector(byzantine_vector, gradient_clip)
        byzantine_vectors = [byzantine_vector] * nb_byz
        distance = aggregator.aggregate(temp_vectors + byzantine_vectors).sub(avg_honest_vector)
        return distance.norm().item()
    best_factor = misc.line_maximize(eval_factor_FOE)
    return fall_of_empires(honest_vectors, attack_factor=best_factor)


def a_little_is_enough(honest_vectors, attack_factor=1.5, negative=False, **kwargs):
    #JS: negative controls the sign of the attack_factor
    #JS: attack_factor controls the magnitude of the attack_vector to be added to avg_honest_vector
    stacked_vectors = torch.stack(honest_vectors)
    attack_vector = stacked_vectors.var(dim=0).sqrt_()
    if negative:
        attack_factor = - attack_factor
    return stacked_vectors.mean(dim=0).add(attack_vector, alpha=attack_factor)


def auto_ALIE(honest_vectors, aggregator, nb_byz, gradient_clip, **kwargs):
    avg_honest_vector = torch.stack(honest_vectors).mean(dim=0)
    def eval_factor_ALIE(factor):
        #temp_vectors = copy.deepcopy(honest_vectors)
        temp_vectors = [vector.clone() for vector in honest_vectors]
        byzantine_vector = a_little_is_enough(temp_vectors, attack_factor=factor)
        if gradient_clip is not None:
            byzantine_vector = misc.clip_vector(byzantine_vector, gradient_clip)
        byzantine_vectors = [byzantine_vector] * nb_byz
        distance = aggregator.aggregate(temp_vectors + byzantine_vectors).sub(avg_honest_vector)
        return distance.norm().item()
    best_factor = misc.line_maximize(eval_factor_ALIE)
    return a_little_is_enough(honest_vectors, attack_factor=best_factor)


def fang_TMA(honest_vectors, b=2.0, **kwargs):
    """
    Fang Attack adapted for Model Aggregation (Direct weights).
    Generates a malicious model vector based on the statistics of honest models.

    Args:
        honest_vectors (torch.Tensor): A tensor of shape (num_honest, d) containing 
                                       the flattened parameters of all honest clients.
        b (float): Attack scaling factor (default 2.0).

    Returns:
        torch.Tensor: A single malicious vector of shape (d,).
    """
    # 1. Calculate statistics across honest clients (column-wise)
    # b_max: The maximum value for each parameter across honest clients
    # b_min: The minimum value for each parameter across honest clients
    honest_vectors = torch.stack(honest_vectors)
    b_max, _ = torch.max(honest_vectors, dim=0)
    b_min, _ = torch.min(honest_vectors, dim=0)
    
    # average_sign: The direction of the consensus (sign of the mean)
    avg_model = torch.mean(honest_vectors, dim=0)
    avg_sign = torch.sign(avg_model)

    # 2. Generate Randomness
    # The attack picks a random point within a specific extended range
    rand_factors = torch.rand_like(avg_model) # Uniform [0, 1]

    # 3. Vectorized Logic (matching the heuristic exactly)
    
    # --- Logic for Negative Average Direction ---
    # If the consensus is negative, we want to push towards the negative tail (b_min).
    # If b_min is positive (rare but possible), we push towards 0 (b_min/b).
    # If b_min is negative, we push further negative (b_min * b).
    target_min_scaled = torch.where(b_min > 0, b_min / b, b_min * b)
    # Interpolate random value between [target, b_min]
    val_neg = target_min_scaled + (b_min - target_min_scaled) * rand_factors

    # --- Logic for Positive Average Direction ---
    # If the consensus is positive, we want to push towards the positive tail (b_max).
    # If b_max is positive, we push further positive (b_max * b).
    # If b_max is negative (rare), we push towards 0 (b_max/b).
    target_max_scaled = torch.where(b_max > 0, b_max * b, b_max / b)
    # Interpolate random value between [b_max, target]
    val_pos = b_max + (target_max_scaled - b_max) * rand_factors

    # 4. Combine based on sign
    # Use the negative logic where avg_sign < 0, and positive logic otherwise
    mal_vector = torch.where(avg_sign < 0, val_neg, val_pos)

    return mal_vector

def mimic(attack, honest_vectors, current_step, **kwargs):
    if attack.learning_phase_mimic is None:
        #JS: Always return the vector of the first worker
        return honest_vectors[0]

    if current_step < attack.learning_phase_mimic:
        #JS: Update mu_mimic and z_mimic only if still in learning phase
        attack.update_mimic_heuristic(honest_vectors, current_step)
        #JS: Return the vector of the first worker in learning phase
        return honest_vectors[0]

    current_max = -1
    best_worker_to_mimic = None
    for i, vector in enumerate(honest_vectors):
        dot_product = torch.dot(vector, attack.z_mimic).norm().item()
        if dot_product > current_max:
            current_max = dot_product
            best_worker_to_mimic = i

    return honest_vectors[best_worker_to_mimic]


#JS: infinity attack
def inf(attack, **kwargs):
    return torch.ones(attack.model_size, device=attack.device) * float('inf')

#JS: Dictionary mapping every Byzantine attack to its corresponding function
byzantine_attacks = {"SF": signflipping, "LF": labelflipping, "FOE": fall_of_empires, "ALIE": a_little_is_enough, "mimic": mimic,
                    "auto_ALIE": auto_ALIE, "auto_FOE": auto_FOE, "inf": inf, "TMA": fang_TMA}

class ByzantineAttack(object):

    def __init__(self, attack_name, nb_real_byz, model_size, device, learning_phase, gradient_clip, robust_aggregator):
        self.attack_name = attack_name
        #JS: nb_real_byz is the actual number of "real" Byzantine workers
        self.nb_real_byz = nb_real_byz
        #JS: Instantiate the robust aggregator to be used (in particular to be used for auto ALIE and auto FOE)
        self.robust_aggregator = robust_aggregator
        self.model_size = model_size
        self.gradient_clip = gradient_clip
        self.device = device
        #JS: parameters for the mimic_heuristic attack
        self.z_mimic = torch.rand(model_size, device=device)
        self.mu_mimic = torch.zeros(model_size, device=device)
        self.learning_phase_mimic = learning_phase


    def generate_byzantine_vectors(self, honest_vectors, flipped_vectors, current_step):
        if self.nb_real_byz == 0:
            return list()

        #JS: Compute the Byzantine vectors
        byzantine_vector = byzantine_attacks[self.attack_name](attack=self, honest_vectors=honest_vectors, flipped_vectors=flipped_vectors, learning_phase=self.learning_phase_mimic,
                                                   current_step=current_step, aggregator=self.robust_aggregator, nb_byz = self.nb_real_byz, gradient_clip=self.gradient_clip)
        if self.gradient_clip is not None:
            byzantine_vector = misc.clip_vector(byzantine_vector, self.gradient_clip)

        return [byzantine_vector] * self.nb_real_byz


    #JS: update the parameters of the mimic attack
    def update_mimic_heuristic(self, honest_vectors, current_step):
        time_factor = 1 / (current_step + 2)
        step_ratio = (current_step + 1) * time_factor
        self.mu_mimic.mul_(step_ratio)
        self.mu_mimic.add_(torch.stack(honest_vectors).mean(dim=0), alpha=time_factor)

        self.z_mimic.mul_(step_ratio)
        cumulative = torch.zeros(self.model_size, device=self.device)
        for vector in honest_vectors:
            deviation = torch.sub(vector, self.mu_mimic)
            deviation.mul_(torch.dot(deviation, self.z_mimic).norm().item())
            cumulative.add_(deviation)

        self.z_mimic.mul_(step_ratio)
        self.z_mimic.add_(torch.nn.functional.normalize(cumulative, dim=0), alpha=time_factor)
        self.z_mimic.div_(self.z_mimic.norm().item())