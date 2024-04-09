import timm
import torch.nn as nn
import torch.nn.functional as F

class Xception(nn.Module):
    def __init__(self, num_classes, pretrain=True):
        super(Xception, self).__init__()
        if pretrain:
            self.xception = timm.create_model('xception', pretrained=True, num_classes=num_classes)
        else:
            self.xception = timm.create_model('xception', pretrained=False, num_classes=num_classes)

        # Modify the final classification layer for the number of classes in CIFAR-10

    def load_state_dict(self, state_dict):
        self.xception.load_state_dict(state_dict)

    def forward(self, x):
        return self.xception(x)
