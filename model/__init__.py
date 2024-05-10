from typing import Dict
from utils.params import init_kaiming_normal

import torch.nn as nn
from enums import MODELS
from .Resnet18 import Resnet18
from .Resnet34 import Resnet34
from .Xception import Xception

models: Dict[str, nn.Module] = {
    MODELS.RESNET18: Resnet18,
    MODELS.RESNET34: Resnet34,
    MODELS.XCEPTION: Xception
}


def get(name: str, num_classes: int, pretrain: bool = False, dropout: float = 0.):
    if pretrain:
        return models[name](num_classes=num_classes, pretrain=True, dropout=dropout)
    else:
        model = models[name](num_classes=num_classes, pretrain=False, dropout=dropout)
        init_kaiming_normal(model)
        return model
