import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet34_Weights
from utils.model import remove_prefix_from_state_dict



class Resnet34(nn.Module):
    def __init__(self, num_classes, pretrain=True, dropout=0.):
        super(Resnet34, self).__init__()
        if pretrain:
            self.resnet34 = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        else:
            self.resnet34 = models.resnet34()

        # Modify the final classification layer for the number of classes in CIFAR-10
        in_features = self.resnet34.fc.in_features

        if dropout > 0.0001:
            self.resnet34.fc = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes, bias=True)
            )
        else:
            self.resnet34.fc = nn.Linear(in_features, num_classes, bias=True)


    def load_state_dict(self, state_dict):
        state_dict = remove_prefix_from_state_dict(state_dict, prefix_to_remove='resnet34.')
        state_dict['fc.weight'] = state_dict.pop('fc.1.weight')
        state_dict['fc.bias'] = state_dict.pop('fc.1.bias')
        self.resnet34.load_state_dict(state_dict)

    def forward(self, x):
        return self.resnet34(x)
