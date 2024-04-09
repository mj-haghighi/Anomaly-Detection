import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet34_Weights



class Resnet34(nn.Module):
    def __init__(self, num_classes, pretrain=True):
        super(Resnet34, self).__init__()
        if pretrain:
            self.resnet = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        else:
            self.resnet = models.resnet34()

        # Modify the final classification layer for the number of classes in CIFAR-10
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)

    def load_state_dict(self, state_dict):
        self.resnet18.load_state_dict(state_dict)

    def forward(self, x):
        return self.resnet(x)
