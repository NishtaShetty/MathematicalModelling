"""
data/partition.py
Loads MNIST and splits it across N clients using IID or Non-IID strategy.
Non-IID is realistic for federated learning papers.
"""

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from collections import defaultdict
import random
import numpy as np


def load_mnist(data_dir="./data/raw"):
    """Download and return MNIST train/test datasets."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(data_dir, train=True,  download=True, transform=transform)
    test_dataset  = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    return train_dataset, test_dataset


def iid_partition(dataset, num_clients, samples_per_client=500):
    """
    IID partition: each client gets a random uniform sample.
    """
    all_indices = list(range(len(dataset)))
    random.shuffle(all_indices)
    client_data = {}
    for i in range(num_clients):
        start = i * samples_per_client
        end   = start + samples_per_client
        client_data[i] = all_indices[start:end]
    return client_data


def non_iid_partition(dataset, num_clients, classes_per_client=2, samples_per_class=250):
    """
    Non-IID partition: each client only sees `classes_per_client` out of 10 classes.
    This creates realistic data heterogeneity across clients.

    Args:
        dataset: torchvision MNIST dataset
        num_clients: number of FL clients
        classes_per_client: how many distinct digit classes each client holds
        samples_per_class: samples drawn per assigned class

    Returns:
        dict {client_id: [list of dataset indices]}
    """
    # Group indices by label
    label_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        label_indices[label].append(idx)

    # Shuffle within each class
    for label in label_indices:
        random.shuffle(label_indices[label])

    client_data = {}
    for client_id in range(num_clients):
        # Assign classes cyclically to ensure coverage
        assigned_classes = [(client_id * classes_per_client + k) % 10
                            for k in range(classes_per_client)]
        indices = []
        for cls in assigned_classes:
            pool = label_indices[cls]
            n = min(samples_per_class, len(pool))
            indices.extend(pool[:n])
        random.shuffle(indices)
        client_data[client_id] = indices

    return client_data


def get_client_loaders(dataset, client_data, batch_size=32):
    """
    Convert index lists to DataLoaders for each client.

    Returns:
        dict {client_id: DataLoader}
    """
    loaders = {}
    for cid, indices in client_data.items():
        subset = Subset(dataset, indices)
        loaders[cid] = DataLoader(subset, batch_size=batch_size, shuffle=True)
    return loaders


def get_test_loader(test_dataset, batch_size=256):
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def label_flip_loader(dataset, indices, source_label=1, target_label=7, batch_size=32):
    """
    Returns a DataLoader where `source_label` samples are relabelled as `target_label`.
    Used by adversarial clients performing label-flipping attacks.
    """
    class FlippedSubset(torch.utils.data.Dataset):
        def __init__(self, dataset, indices, src, tgt):
            self.dataset = dataset
            self.indices = indices
            self.src = src
            self.tgt = tgt

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, i):
            x, y = self.dataset[self.indices[i]]
            if y == self.src:
                y = self.tgt
            return x, y

    flipped = FlippedSubset(dataset, indices, source_label, target_label)
    return DataLoader(flipped, batch_size=batch_size, shuffle=True)
