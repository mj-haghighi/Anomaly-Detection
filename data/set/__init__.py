from typing import Dict
from enums import DATASETS
from data.set.dataset_interface import IDataset

from .mnist import Dataset as MNISTDataset

datasets: Dict[str, IDataset] = {
    DATASETS.mnist: MNISTDataset
}