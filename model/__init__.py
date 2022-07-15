from typing import Dict
import torch.nn as nn
from enums import DATASETS
from .mnist import MNIST

models: Dict[str, nn.Module]={
    DATASETS.mnist: MNIST
}
