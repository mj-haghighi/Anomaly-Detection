from typing import Dict
from enums import DATASETS
from data.set.dataset_interface import IDataset

from .mnist import Dataset as MNISTDataset
from .cifar10 import Dataset as CIFAR10Dataset

datasets: Dict[str, IDataset] = {
    DATASETS.mnist: MNISTDataset,
    DATASETS.cifar10: CIFAR10Dataset
}