from typing import Dict
from utils.params import init_kaiming_normal

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


def get(name: str, num_classes: int, pretrain: bool = False):
    if pretrain:
        return models[name](num_classes=num_classes, pretrain=True)
    else:
        model = models[name](num_classes=num_classes, pretrain=False)
        init_kaiming_normal(model)
        return model
