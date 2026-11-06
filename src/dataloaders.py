import torch
import torchvision.transforms as transforms
import numpy as np
from numpy import genfromtxt
import random
from torchvision import datasets, transforms
from sklearn import datasets as sk_datasets
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_svmlight_file
from torch.utils.data import Dataset
import sys
from tqdm import tqdm, trange
#from src.svm import get_phishing
from torch.utils.data.sampler import SubsetRandomSampler
import logging
import os



def homogeneous_mnist_train_test(config):
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]))

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Calculate the number of samples per client
    num_clients = config['n'] - config['f']
    samples_per_client = len(dataset) // num_clients

    dataloaders_dict = {}
    for i in range(num_clients):
        # Create a subset of the dataset for this client
        client_dataset = torch.utils.data.Subset(dataset, range(i * samples_per_client, (i + 1) * samples_per_client))

        # Split the client dataset into train and test sets
        train_size = int(0.8 * len(client_dataset))  # 80% for training
        test_size = len(client_dataset) - train_size
        client_train_dataset, client_test_dataset = torch.utils.data.random_split(client_dataset, [train_size, test_size])

        # Create dataloaders for train and test sets
        client_train_loader = torch.utils.data.DataLoader(client_train_dataset, batch_size=32, shuffle=True)
        client_test_loader = torch.utils.data.DataLoader(client_test_dataset, batch_size=128, shuffle=False)

        dataloaders_dict[i] = [client_train_loader, client_test_loader]

    return dataloaders_dict


