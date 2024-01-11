import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet18_Weights


class Resnet18(nn.Module):
    def __init__(self, num_classes, pretrain=True):
        super(Resnet18, self).__init__()
        if pretrain:
            self.resnet18 = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.resnet18 = models.resnet18()

        # Modify the final classification layer for the number of classes in CIFAR-10
        in_features = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(in_features, num_classes)

    def load_state_dict(self, state_dict):
        self.resnet18.load_state_dict(state_dict)

    def forward(self, x):
        return self.resnet18(x)
