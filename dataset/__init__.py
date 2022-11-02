import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler 

from . preprocess import DataPreprocessor
from config import INTERNAL_DTYPE

from dataclasses import dataclass


@dataclass
class Dataset:
    """
    Contains all dataset related data.
    """
    train: TensorDataset
    test: TensorDataset
    num_train_data: int
    num_features: int
    num_labels: int


def load_dataset(filename, test_size=0.2):
    preprocessor = DataPreprocessor(test_size)

    X_train, X_test, Y_train, Y_test = preprocessor.fit_load(filename)

    num_train_data = X_train.shape[0]
    num_features = X_train.shape[1]
    num_labels = len(preprocessor.y_transformer.classes_)

    train_dataset = TensorDataset(
            torch.from_numpy(X_train).to(INTERNAL_DTYPE.torch),
            torch.from_numpy(Y_train))
    test_dataset = TensorDataset(
            torch.from_numpy(X_test).to(INTERNAL_DTYPE.torch),
            torch.from_numpy(Y_test))

    return Dataset(
            train_dataset, test_dataset,
            num_train_data, num_features,
            num_labels,
            )


# Divide data to users
def split_data_equal(dataset, groups, BATCH_SIZE):
    """
    Split the given dataset to the given number of equal parts.
    Returns a list of dataloaders.
    """
    chunk_size = len(dataset) // groups
    # group_indices = []
    dataloaders = []
    remaining = np.arange(len(dataset))
    for i in range(groups):
        indices = np.random.choice(remaining, chunk_size, replace=False)
        sampler = SubsetRandomSampler(indices)
        dataloaders.append(DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, sampler=sampler))
        remaining = list(set(remaining) - set(indices))
    return dataloaders


