import torch, math, argparse, sys, os
from itertools import combinations
from . import study
import pandas, numpy, tools, pathlib
import numpy as np

# ---------------------------------------------------------------------------- #
#JS: Functions used for experiments in the reproduce scripts

#JS: Function used to create the directories needed to store ther results in the reproduce scripts
def check_make_dir(path):
  path = pathlib.Path(path)
  if path.exists():
    if not path.is_dir():
      tools.fatal(f"Given path {str(path)!r} must point to a directory")
  else:
    path.mkdir(mode=0o755, parents=True)
  return path

#JS: Function used to parse the command-line and perform checks in the reproduce scripts
def process_commandline():
  parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
  parser.add_argument("--result-directory",
    type=str,
    default="results-data",
    help="Path of the data directory, containing the data gathered from the experiments")
  parser.add_argument("--plot-directory",
    type=str,
    default="results-plot",
    help="Path of the plot directory, containing the graphs traced from the experiments")
  parser.add_argument("--devices",
    type=str,
    default="auto",
    help="Comma-separated list of devices on which to run the experiments, used in a round-robin fashion")
  parser.add_argument("--supercharge",
    type=int,
    default=1,
    help="How many experiments are run in parallel per device, must be positive")
  # Parse command line
  return parser.parse_args(sys.argv[1:])


#JS: Function used to plot the results of the experiments in the reproduce scripts
def compute_avg_err_op(name, seeds, result_directory, location, *colops, avgs="", errs="-err"):
  """ Compute the average and standard deviation of the selected columns over the given experiment.
  Args:
    name Given experiment name
    seeds   Seeds used for the experiment
    result_directory Directory to store the results
    location Script to read from
    ...  Tuples of (selected column name (through 'study.select'), optional reduction operation name)
    avgs Suffix for average column names
    errs Suffix for standard deviation (or "error") column names
  Returns:
    Data frames for each of the computed columns,
    Tuple of reduced values per seed (or None if None was provided for 'op')
  Raises:
    'RuntimeError' if a reduction operation was specified for a column selector that did not select exactly 1 column
  """
# Load all the runs for the given experiment name, and keep only a subset
  datas = tuple(study.select(study.Session(result_directory + "/" + name + "-" +str(seed), location), *(col for col, _ in colops)) for seed in seeds)

  # Make the aggregated data frames
  def make_df_ro(col, op):
    nonlocal datas
    # For every selected columns
    subds = tuple(study.select(data, col).dropna() for data in datas)
    df    = pandas.DataFrame(index=subds[0].index)
    ro    = None
    for cn in subds[0]:
      # Generate compound column names
      avgn = cn + avgs
      errn = cn + errs
      # Compute compound columns
      numds = numpy.stack(tuple(subd[cn].to_numpy() for subd in subds))
      df[avgn] = numds.mean(axis=0)
      df[errn] = numds.std(axis=0)
      # Compute reduction, if requested
      if op is not None:
        if ro is not None:
          raise RuntimeError(f"column selector {col!r} selected more than one column ({(', ').join(subds[0].columns)}) while a reduction operation was requested")
        ro = tuple(getattr(subd[cn], op)().item() for subd in subds)
    # Return the built data frame and optional computed reduction
    return df, ro
  dfs = list()
  ros = list()
  for col, op in colops:
    df, ro = make_df_ro(col, op)
    dfs.append(df)
    ros.append(ro)
  # Return the built data frames and optional computed reductions
  return dfs


# ---------------------------------------------------------------------------- #
#JS: Functions used for dataset manipulation in dataset.py

#JS: Lazy-initialize and return the default dataset root directory path
def get_default_root():
    # Generate the default path
    default_root = pathlib.Path(__file__).parent / "datasets" / "cache"
    # Create the path if it does not exist
    default_root.mkdir(parents=True, exist_ok=True)
    # Return the path
    return default_root

#JS: Returns the indices of the training datapoints selected for each honest worker, in case of Dirichlet distribution
def draw_indices(samples_distribution, indices_per_label, nb_workers):
    
    #JS: Initialize the dictionary of samples per worker. Should hold the indices of the samples each worker possesses
    worker_samples = dict()
    for worker in range(nb_workers):
        worker_samples[worker] = list()

    for label, label_distribution in enumerate(samples_distribution):
        last_sample = 0
        number_samples_label = len(indices_per_label[label])
        #JS: Iteratively split the number of samples of label into chunks according to the worker proportions, and assign each chunk to the corresponding worker
        for worker, worker_proportion in enumerate(label_distribution):
            samples_for_worker = int(worker_proportion * number_samples_label)
            worker_samples[worker].extend(indices_per_label[label][last_sample:last_sample+samples_for_worker])
            last_sample = samples_for_worker

    return worker_samples


# ---------------------------------------------------------------------------- #
#JS: Functions used in train.py and train_p2p.py

#JS: Store a result in the corresponding list result file.
def store_result(fd, *entries):
	"""
	Args:
		fd     Descriptor of the valid result file
		entries... Object(s) to convert to string and write in order in a new line
	"""
	fd.write(os.linesep + ("\t").join(str(entry) for entry in entries))
	fd.flush()

#JS: Create the results file.
def make_result_file(fd, *fields):
	"""
	Args:
		fd     Descriptor of the valid result file
		entries... Object(s) to convert to string and write in order in a new line
	"""
	fd.write("# " + ("\t").join(str(field) for field in fields))
	fd.flush()

#JS: Print the configuration of the current training in question
def print_conf(subtree, level=0):
  if isinstance(subtree, tuple) and len(subtree) > 0 and isinstance(subtree[0], tuple) and len(subtree[0]) == 2:
    label_len = max(len(label) for label, _ in subtree)
    iterator  = subtree
  elif isinstance(subtree, dict):
    if len(subtree) == 0:
      return " - <none>"
    label_len = max(len(label) for label in subtree.keys())
    iterator  = subtree.items()
  else:
    return f" - {subtree}"
  level_spc = "  " * level
  res = ""
  for label, node in iterator:
    res += f"{os.linesep}{level_spc}· {label}{' ' * (label_len - len(label))}{print_conf(node, level + 1)}"
  return res


# ---------------------------------------------------------------------------- #
#JS: Criterions to evaluate accuracy of models. Used in worker.py and p2pWorker.py

def topk(output, target, k=1):
      """ Compute the top-k criterion from the output and the target.
      Args:
        output Batch × model logits
        target Batch × target index
      Returns:
        1D-tensor [#correct classification, batch size]
      """
      res = (output.topk(k, dim=1)[1] == target.view(-1).unsqueeze(1)).any(dim=1).sum()
      return torch.cat((res.unsqueeze(0), torch.tensor(target.shape[0], dtype=res.dtype, device=res.device).unsqueeze(0)))


def sigmoid(output, target):
      """ Compute the sigmoid criterion from the output and the target.
      Args:
        output Batch × model logits (expected in [0, 1])
        target Batch × target index (expected in {0, 1})
      Returns:
        1D-tensor [#correct classification, batch size]
      """
      correct = target.sub(output).abs_() < 0.5
      res = torch.empty(2, dtype=output.dtype, device=output.device)
      res[0] = correct.sum()
      res[1] = len(correct)
      return res

# ---------------------------------------------------------------------------- #
#JS: Functions for manipulating gradients and model parameters

#JS: Clip the vector if its L2 norm is greater than clip_threshold
def clip_vector(vector, clip_threshold):
    vector_norm = vector.norm().item()
    if vector_norm > clip_threshold:
        vector.mul_(clip_threshold / vector_norm)
    return vector

#JS: flatten list of tensors. Used for model parameters and gradients
def flatten(list_of_tensors):
    return torch.cat(tuple(tensor.view(-1) for tensor in list_of_tensors))

#JS: unflatten a flat tensor. Used when setting model parameters and gradients
def unflatten(flat_tensor, model_shapes):
    c = 0
    returned_list = [torch.zeros(shape) for shape in model_shapes]
    for i, shape in enumerate(model_shapes):
        count = 1
        for element in shape:
            count *= element
        returned_list[i].data = flat_tensor[c:c + count].view(shape)
        c = c + count
    return returned_list

# ---------------------------------------------------------------------------- #
#JS: Functions for robust aggregators

#JS: Approximation algorithm used for geometric median
def smoothed_weiszfeld(nb_vectors, vectors, nu=0.1, T=3):
    z = torch.zeros_like(vectors[0])
    # Calculate mask to exclude vectors containing infinite values
    mask = ~torch.any(torch.isinf(torch.stack(vectors)), dim=-1)
    filtered_vectors = [v for v, m in zip(vectors, mask) if m]
    alphas = [1 / nb_vectors] * len(filtered_vectors)
    for _ in range(T):
        betas = list()
        for i, vector in enumerate(filtered_vectors):
            distance = z.sub(vector).norm().item()
            if math.isnan(distance):
                # Distance is infinite or NaN
                betas.append(0)
            else:
                betas.append(alphas[i] / max(distance, nu))
        z.zero_()
        for vector, beta in zip(filtered_vectors, betas):
            z.add_(vector, alpha=beta)
        z.div_(sum(betas))
    return z


def smoothed_weiszfeld2(nb_vectors, vectors, nu=0.1, T=3):
    """ Smoothed Weiszfeld algorithm
    Args:
        vectors: non-empty list of vectors to aggregate
        alphas: scaling factors
        nu: RFA parameter
        T: number of iterations to run the smoothed Weiszfeld algorithm
    Returns:
        Aggregated vector
    """
    z = torch.zeros_like(vectors[0])
    vectors = torch.stack(vectors)
    # Exclude vectors that contain any infinite values
    mask = ~torch.any(torch.isinf(vectors), dim=-1)
    filtered_vectors = vectors[mask]
    alphas = torch.tensor([1 / nb_vectors] * len(filtered_vectors)).to(vectors[0].device)
    for _ in range(T):
        distances = torch.linalg.vector_norm(z - filtered_vectors, dim=-1)
        betas = torch.div(alphas, torch.clamp(distances, min=nu))
        # Update z using the betas and filtered vectors
        z = torch.sum(betas[:, None] * filtered_vectors, dim=0).div(betas.sum())
    return z


#JS: used for Krum, Multi-Krum, and NNM aggregators
def compute_distances(vectors):
    #Compute all pairwise distances between vectors in a list
    if type(vectors) != torch.Tensor:
       vectors = torch.stack(vectors)
    distances = torch.cdist(vectors, vectors)#, compute_mode='use_mm_for_euclid_dist')
    # set non-finite values to inf
    distances[~torch.isfinite(distances)] = float('inf')
    return distances


#JS: Get the vector with the smallest score for Krum aggregator
def get_vector_best_score(vectors, nb_byz, distances):
    vectors = torch.stack(vectors)
    n_vectors = vectors.size(0)
    min_score, min_index = torch.tensor(math.inf), 0
    for worker_id in range(n_vectors):
        # Create a mask for selecting all vectors except the current one
        mask = torch.ones(n_vectors, dtype=torch.bool)
        mask[worker_id] = 0
        # Select all distances to the current vector
        distances_to_vector = distances[worker_id, mask]
        # Square and sort the distances
        distances_squared_to_vector = distances_to_vector.pow(2).sort()[0]
        # Compute the score
        score = distances_squared_to_vector[:n_vectors - nb_byz - 1].sum()
        #score = sum(distances_squared_to_vector[:n_vectors - nb_byz - 1])
        # Update min score and min index
        if score < min_score:
            min_score, min_index = score, worker_id
    # Return the vector with smallest score
    return vectors[min_index]


#JS: get the scores of vectors sorted increasingly, for Multi-Krum
def get_vector_scores(vectors, nb_byz, distances):
    # if type(vectors) != torch.Tensor:
    vectors = torch.stack(vectors)
    n_vectors = vectors.size(0)
    scores = []
    for worker_id in range(n_vectors):
        # Create a mask for selecting all vectors except the current one
        mask = torch.ones(n_vectors, dtype=torch.bool)
        mask[worker_id] = 0
        # Select all distances to the current vector
        distances_to_vector = distances[worker_id, mask]
        # Square and sort the distances
        distances_squared_to_vector = distances_to_vector.pow(2).sort()[0]
        # Compute the score
        score = distances_squared_to_vector[:n_vectors - nb_byz - 1].sum()
        # Save the score and worker id
        scores.append((score.item(), worker_id))
    # Sort the scores in increasing order
    scores.sort(key=lambda x: x[0])
    return scores


#JS: Compute the average of the n-f closest vectors to pivot
def average_nearest_neighbors(vectors, nb_byz):
    distances = compute_distances(vectors)
    _, indices = torch.sort(distances, dim=1)
    # JS: Return the average of the n-f closest vectors to each vector
    closest_vectors = vectors[indices[:, :(vectors.size(0) - nb_byz)]]
    return closest_vectors.mean(dim=1)


#JS: used for MDA and MVA aggregators
def compute_distances_mda(vectors, nb_vectors):
    """Compute all pairwise distances between vectors"""
    distances = dict()
    all_pairs = list(combinations(range(nb_vectors), 2))
    for (x,y) in all_pairs:
        dist = vectors[x].sub(vectors[y]).norm().item()
        if math.isnan(dist):
            dist = float('inf')
        distances[(x,y)] = dist
    return distances


#JS: Compute the subset of (n-f) gradients of minimum diameter 
def compute_min_diameter_subset(vectors, nb_vectors, nb_byz):
    #JS: compute all pairwise distances
    distances = compute_distances_mda(vectors, nb_vectors)
    min_diameter = float('inf')

    #JS: Get all subsets of size n - f
    all_subsets = list(combinations(range(nb_vectors), nb_vectors - nb_byz))
    for subset in all_subsets:
        subset_diameter = 0
        #JS: Compute diameter of subset
        for i, vector1 in enumerate(subset):
            for vector2 in subset[i+1:]:
                distance = distances.get((vector1, vector2), 0)
                subset_diameter = distance if distance > subset_diameter else subset_diameter

        #JS: Update min diameter (if needed)
        if min_diameter > subset_diameter:
            min_diameter = subset_diameter
            min_subset = subset

    return min_subset


#JS: Compute the subset (indices of vectors) of (n-f) vectors of minimum variance 
def compute_min_variance_subset(vectors, nb_vectors, nb_byz):
    #JS: compute all pairwise distances
    distances = compute_distances_mda(vectors, nb_vectors)

    #JS: Get all subsets of size n - f
    all_subsets = list(combinations(range(nb_vectors), nb_vectors - nb_byz))
    min_variance = float('inf')

    for subset in all_subsets:
        current_variance = 0
        #JS: Compute diameter of subset
        for i, vector1 in enumerate(subset):
            for vector2 in subset[i+1:]:
                distance = distances.get((vector1, vector2), 0)
                current_variance += distance**2
        
        if min_variance > current_variance:
            min_variance = current_variance
            min_subset = subset

    return min_subset


#JS: Compute the n-f closest vectors to the honest vector in question (last element of vectors)
#JS: Used in MoNNA aggregation rule
def compute_closest_vectors_and_mean(vectors, nb_vectors, nb_byz):
    # Convert vectors from a list of 1D tensors to a 2D tensor
    vectors = torch.stack(vectors)
    pivot_vector = vectors[-1]
    # Calculate distances using vectorized operations
    distances = torch.norm(vectors - pivot_vector, dim=1)
    # Get the indices of the smallest n-f distances
    _, indices = torch.topk(distances, k=nb_vectors-nb_byz, largest=False)
    # Use advanced indexing to select the closest vectors and compute their mean
    return vectors[indices].mean(dim=0)


# ---------------------------------------------------------------------------- #
#JS: Functions for Byzantine attacks

#JS: used for Auto ALIE and Auto FOE
def line_maximize(scape, evals=16, start=0., delta=1., ratio=0.8):
  """ Best-effort arg-maximize a scape: ℝ⁺⟶ ℝ, by mere exploration.
  Args:
    scape Function to best-effort arg-maximize
    evals Maximum number of evaluations, must be a positive integer
    start Initial x evaluated, must be a non-negative float
    delta Initial step delta, must be a positive float
    ratio Contraction ratio, must be between 0.5 and 1. (both excluded)
  Returns:
    Best-effort maximizer x under the evaluation budget
  """
  # Variable setup
  best_x = start
  best_y = scape(best_x)
  evals -= 1
  # Expansion phase
  while evals > 0:
    prop_x = best_x + delta
    prop_y = scape(prop_x)
    evals -= 1
    # Check if best
    if prop_y > best_y:
      best_y = prop_y
      best_x = prop_x
      delta *= 2
    else:
      delta *= ratio
      break
  # Contraction phase
  while evals > 0:
    if prop_x < best_x:
      prop_x += delta
    else:
      x = prop_x - delta
      while x < 0:
        x = (x + prop_x) / 2
      prop_x = x
    prop_y = scape(prop_x)
    evals -= 1
    # Check if best
    if prop_y > best_y:
      best_y = prop_y
      best_x = prop_x
    # Reduce delta
    delta *= ratio
  # Return found maximizer
  return best_x

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
