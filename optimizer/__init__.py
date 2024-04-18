from torch.nn import Module
from torch import optim

from enums import OPTIMIZER


def get(name: str = "adam", model: Module = None, learning_rate: float = 0.0001):
    if OPTIMIZER.adam == name:
        return optim.Adam(model.parameters(), lr=learning_rate)
    
    if OPTIMIZER.rmsprobe == name:
        return optim.RMSprop(model.parameters(), lr=learning_rate)
    
    if OPTIMIZER.sgd == name:
        return optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    
    if OPTIMIZER.sparseadam == name:
        return optim.SparseAdam(model.parameters(), lr=learning_rate)
