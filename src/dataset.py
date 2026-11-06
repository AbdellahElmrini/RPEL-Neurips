# coding: utf-8
###
 # @file   dataset.py
 # @author John stephan <john.stephan@epfl.ch>
 #
 # @section LICENSE
 #
 # Copyright © 2023 École Polytechnique Fédérale de Lausanne (EPFL).
 # All rights reserved.
 #
 # @section DESCRIPTION
 #
 # Dataset wrappers/helpers.
###

import torch, torchvision, random
import torchvision.transforms as T
import numpy as np
from . import misc

# ---------------------------------------------------------------------------- #
# Collection of default transforms
transforms_horizontalflip = T.Compose([T.RandomHorizontalFlip(), T.ToTensor()])
# Transforms from "A Little is Enough" (https://github.com/moranant/attacking_distributed_learning)
transforms_mnist = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
# Transforms from https://github.com/kuangliu/pytorch-cifar
transforms_cifar = T.Compose([T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

transform_train = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = T.Compose([
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Per-dataset image transformations (automatically completed, see 'Dataset._get_datasets')
transforms = {
  "mnist":        (transforms_mnist, transforms_mnist),
  "fashionmnist": (transforms_horizontalflip, transforms_horizontalflip),
  "cifar10":      (transform_train, transform_test),
  "cifar100":     (transforms_cifar, transforms_cifar),
  "imagenet":     (transforms_horizontalflip, transforms_horizontalflip) }

#JS: Dataset names in pytorch
dict_names = {
  "mnist":        "MNIST",
  "fashionmnist": "FashionMNIST",
  "emnist":       "EMNIST",
  "cifar10":      "CIFAR10",
  "cifar100":     "CIFAR100",
  "imagenet":     "ImageNet"}

# ---------------------------------------------------------------------------- #
# Dataset wrapper class
class Dataset:
  """ Dataset wrapper class."""

  def __init__(self, dataset_name, heterogeneity=False, numb_labels=None, distinct_datasets=False,
               gamma_similarity=None, alpha_dirichlet=None, nb_datapoints=None, honest_workers=None, batch_size=None):
    """ Training Dataset builder constructor.
    Args:
      dataset_name          Dataset string name
      heterogeneity         Boolean that is true in heterogeneous setting
      numb_labels           Number of labels of the dataset in question
      distinct_datasets     Boolean that is true in setting where honest workers must have distinct datasets (e.g., privacy setting)
      gamma_similarity      Float for distributing the datasets among honest workers
      alpha_dirichlet       Value of parameter alpha for dirichlet distribution
      nb_datapoints         Number of datapoints per honest worker in case of distinct datasets
      honest_workers        Number of honest workers in the system
      batch_size            Batch size used during the training or testing
    """

    #JS: Load the initial training dataset
    dataset = getattr(torchvision.datasets, dict_names[dataset_name])(root=misc.get_default_root(), train=True, download=True,
                                                                      transform=transforms[dataset_name][0])
    targets = dataset.targets
    if isinstance(targets, list):
      targets = torch.FloatTensor(targets)

    #JS: extreme heterogeneity setting while training
    if heterogeneity:
      labels = range(numb_labels)
      ordered_indices = []
      for label in labels:
        label_indices = (targets == label).nonzero().tolist()
        label_indices = [item for sublist in label_indices for item in sublist]
        ordered_indices += label_indices

      self.dataset_dict = {}
      split_indices = np.array_split(ordered_indices, honest_workers)
      for worker_id in range(honest_workers):
        dataset_modified = torch.utils.data.Subset(dataset, split_indices[worker_id].tolist())
        dataset_worker = torch.utils.data.DataLoader(dataset_modified, batch_size=batch_size, shuffle=True)
        #JS: have one dataset iterator per honest worker
        self.dataset_dict[worker_id] = dataset_worker


    #JS: distinct datasets for honest workers with gamma similarity
    elif distinct_datasets and gamma_similarity is not None:
      numb_samples = len(targets)
      numb_samples_iid = int(gamma_similarity * numb_samples)

      #JS: Sample gamma_similarity % of the dataset, and build homogeneous dataset
      homogeneous_dataset, _ = torch.utils.data.random_split(dataset, [numb_samples_iid, numb_samples - numb_samples_iid])

      #JS: Split the indices of the homogeneous dataset onto the honest workers
      split_indices_homogeneous = np.array_split(homogeneous_dataset.indices, honest_workers)

      #JS: Rearrange the entire dataset by sorted labels
      labels = range(numb_labels)
      ordered_indices = []
      for label in labels:
        label_indices = (targets == label).nonzero().tolist()
        label_indices = [item for sublist in label_indices for item in sublist]
        ordered_indices += label_indices
      #JS: split the (sorted) heterogeneous indices equally among the honest workers
      indices_heterogeneous = [index for index in ordered_indices if index not in homogeneous_dataset.indices]
      split_indices_heterogeneous = np.array_split(indices_heterogeneous, honest_workers)

      self.dataset_dict = {}
      for worker_id in range(honest_workers):
        homogeneous_dataset_worker = torch.utils.data.Subset(dataset, split_indices_homogeneous[worker_id])
        heterogeneous_dataset_worker = torch.utils.data.Subset(dataset, split_indices_heterogeneous[worker_id])
        concat_datasets = torch.utils.data.ConcatDataset([homogeneous_dataset_worker, heterogeneous_dataset_worker])
        dataset_worker = torch.utils.data.DataLoader(concat_datasets, batch_size=batch_size, shuffle=True)
        #JS: have one dataset iterator per honest worker
        self.dataset_dict[worker_id] = dataset_worker


    #JS: distinct datasets for honest workers, homogeneous setting
    elif distinct_datasets:
      numb_samples = len(targets)
      sample_indices = list(range(numb_samples))
      random.shuffle(sample_indices)

      self.dataset_dict = {}
      if nb_datapoints is None:
        #JS: split the whole dataset equally among the honest workers
        split_indices = np.array_split(sample_indices, honest_workers)
      else:
        #JS: give every honest worker nb_datapoints samples
        split_indices = [sample_indices[i:i + nb_datapoints] for i in range(0, nb_datapoints*honest_workers, nb_datapoints)]

      for worker_id in range(honest_workers):
        dataset_modified = torch.utils.data.Subset(dataset, split_indices[worker_id])
        #JS: have one dataset iterator per honest worker
        self.dataset_dict[worker_id] = torch.utils.data.DataLoader(dataset_modified, batch_size=batch_size, shuffle=True)


    #JS: distribute data among honest workers using Dirichlet distribution
    elif alpha_dirichlet is not None:

      #JS: store in indices_per_label the list of indices of each label (0 then 1 then 2 ...)
      indices_per_label = dict()
      for label in range(numb_labels):
        label_indices = (targets == label).nonzero().tolist()
        label_indices = [item for sublist in label_indices for item in sublist]
        indices_per_label[label] = label_indices

      #JS: compute number of samples of each worker for each class, using a Dirichlet distribution of parameter alpha_dirichlet
      samples_distribution = np.random.dirichlet(np.repeat(alpha_dirichlet, honest_workers), size=numb_labels)
      #JS: get the indices of the samples belonging to each worker (stored in dict worker_samples)
      worker_samples = misc.draw_indices(samples_distribution, indices_per_label, honest_workers)

      self.dataset_dict = {}
      for worker_id in range(honest_workers):
        dataset_modified = torch.utils.data.Subset(dataset, worker_samples[worker_id])
        #JS: have one dataset iterator per honest worker
        self.dataset_dict[worker_id] = torch.utils.data.DataLoader(dataset_modified, batch_size=batch_size, shuffle=True)

# ---------------------------------------------------------------------------- #
def make_train_test_datasets(dataset, heterogeneity=False, numb_labels=None, distinct_datasets=False, gamma_similarity=None, alpha_dirichlet=None,
    nb_datapoints=None, honest_workers=None, train_batch=None, test_batch=None):
  """ Helper to make new instance of train and test datasets.
  Args:
    dataset             Case-sensitive dataset name
    heterogeneity       Boolean that is true in heterogeneous setting
    numb_labels         Number of labels of dataset
    distinct_datasets   Boolean that is true in setting where honest workers must have distinct datasets (e.g., privacy setting)
    gamma_similarity    Float for distributing the datasets among honest workers
    alpha_dirichlet     Value of parameter alpha for dirichlet distribution
    nb_datapoints       Number of datapoints per honest worker in case of distinct datasets
    honest_workers      Number of honest workers in the system
    train_batch         Training batch size
    test_batch          Testing batch size
  Returns:
    Dictionary of training datasets for honest workers and data loader for test dataset
  """
  # Make the training dataset
  trainset = Dataset(dataset, heterogeneity=heterogeneity, numb_labels=numb_labels,
                     distinct_datasets=distinct_datasets, gamma_similarity=gamma_similarity, alpha_dirichlet=alpha_dirichlet,
                     nb_datapoints=nb_datapoints, honest_workers=honest_workers, batch_size=train_batch)

  # Make the testing dataset
  dataset_test = getattr(torchvision.datasets, dict_names[dataset])(root=misc.get_default_root(), train=False, download=True,
                                                                              transform=transforms[dataset][1])
  data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=test_batch, shuffle=False)

  # Return the data loaders
  return trainset.dataset_dict, data_loader_test