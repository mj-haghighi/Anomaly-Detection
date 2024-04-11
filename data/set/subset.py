import copy
from torch.utils.data import Subset as TorchSubset
from torch.utils.data import Dataset as TorchDataset
from typing import Sequence

class Subset(TorchSubset):
    def __init__(self, dataset: TorchDataset, indices: Sequence[int], transform = None):
        self.indices = indices
        
        if transform:
            self.dataset = copy.deepcopy(dataset)
            self.dataset.transform = transform
        else:
            self.dataset = datset