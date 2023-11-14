from torch.nn import Module
from torch import optim

from enums import OPTIMIZER


def load(name: str = "adam", model: Module = None, learning_rate: float = 0.0001):
    print('name: ', name)
    if OPTIMIZER.adam == name:
        return optim.Adam(model.parameters(), lr=learning_rate)
    
    if OPTIMIZER.rmsprobe == name:
        return optim.RMSprop(model.parameters(), lr=learning_rate)
    
    print('OPTIMIZER.sgd: ', OPTIMIZER.sgd)
    if OPTIMIZER.sgd == name:
        return optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    
    if OPTIMIZER.sparseadam == name:
        return optim.SparseAdam(model.parameters(), lr=learning_rate)
