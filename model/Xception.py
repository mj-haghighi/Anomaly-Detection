import timm
import torch.nn as nn
import torch.nn.functional as F
from utils.model import remove_prefix_from_state_dict

class Xception(nn.Module):
    def __init__(self, num_classes, pretrain=True, dropout=0.):
        super(Xception, self).__init__()
        if pretrain:
            self.xception = timm.create_model('xception', pretrained=True, num_classes=num_classes)
        else:
            self.xception = timm.create_model('xception', pretrained=False, num_classes=num_classes)

        if dropout > 0.0001:
            self.xception.fc = nn.Sequential(
                nn.Dropout(p=dropout),
                self.xception.get_classifier()
            )

    def load_state_dict(self, state_dict):
        state_dict = remove_prefix_from_state_dict(state_dict, prefix_to_remove='xception.')
        self.xception.load_state_dict(state_dict)

    def forward(self, x):
        return self.xception(x)
