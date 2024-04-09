from typing import Dict
import torch.nn as nn
from enums import MODELS
from .Resnet18 import Resnet18
from .Resnet34 import Resnet34
from .Xception import Xception

models: Dict[str, nn.Module] = {
    MODELS.resnet18: Resnet18,
    MODELS.resnet34: Resnet34,
    MODELS.xception: Xception
}
