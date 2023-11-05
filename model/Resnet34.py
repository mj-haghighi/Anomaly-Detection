import torch.nn as nn
import torch.nn.functional as F


class Resnet34(nn.Module):
    def __init__(self, num_classes):
        super(Resnet34, self).__init__()
        self.resnet = models.resnet34(pretrained=True)

        # Freeze all layers except the final classification layer
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.resnet.fc.requires_grad = True

        # Modify the final classification layer for the number of classes in CIFAR-10
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)
