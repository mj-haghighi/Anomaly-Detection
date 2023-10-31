import torch.nn as nn
import torch.nn.functional as F


class Resnet18(nn.Module):
    def __init__(self, num_classes):
        super(FineTunedResNet18, self).__init__()
        self.resnet18 = models.resnet18(pretrained=True)

        # Freeze all layers except the final classification layer
        for param in self.resnet18.parameters():
            param.requires_grad = False
        self.resnet18.fc.requires_grad = True

        # Modify the final classification layer for the number of classes in CIFAR-10
        in_features = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.resnet18(x)
