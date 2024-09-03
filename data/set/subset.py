import copy
import torch
import numpy as np
from torch.utils.data import Subset as TorchSubset
from torch.utils.data import Dataset as TorchDataset
from typing import Sequence
from enums import PHASE
from PIL import Image
from utils import to_categorical


class Subset(TorchSubset):
    def __init__(self, dataset: TorchDataset, indices: Sequence[int], phase=None, transform=None):
        self.indices = indices
        self.dataset = copy.deepcopy(dataset)

        if phase:
            self.dataset.phase = phase
        if transform:
            self.dataset.transform = transform


class CombinedTrainSubset(TorchSubset):
    def __init__(self, dataset: TorchDataset, indices: Sequence[int], transform, basic_transform):
        self.indices = np.concatenate((indices, indices + dataset.dataset_config.train_size))
        self.dataset = copy.deepcopy(dataset)
        self.dataset.transform = transform

        def getitem(reference, idx):
            index, img_path, label = reference.samples[idx % reference.dataset_length]
            clabel = to_categorical.sample(label, reference.dataset_config.labels)
            clabel = torch.Tensor(clabel)
            img = Image.open(img_path)

            if idx >= reference.dataset_length:
                img = transform(img)
                return str(index) + '_a', img, clabel
            else:
                img = basic_transform(img)
                return str(index) + '_na', img, clabel

        def get_len(reference):
            return 2 * reference.dataset_length

        self.dataset.overide_getitem = getitem
        self.dataset.overide_len = get_len
