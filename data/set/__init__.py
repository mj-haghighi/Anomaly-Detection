from typing import Dict
from enums import DATASETS
from data.set.DatasetInterface import IDataset

from .mnist import Dataset as MNISTDataset

datasets: Dict[str, IDataset] = {
    DATASETS.mnist: MNISTDataset
}