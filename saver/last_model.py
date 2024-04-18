import os
import torch
import os.path as osp
import torch.nn as nn

class LastEpochModelSaver:
    def __init__(self, savedir:str) -> None:
        self.savedir = savedir
    
    def save(self, epoch: int, model: nn.Module, fold: int = None):
        path = self.savedir

        if epoch == 0:
            self.last_model = None

        if fold is not None:
            path = osp.join(path, f'{fold}')

        if not osp.exists(path):
            os.makedirs(path)
        path = osp.join(path, f'ep{epoch}-last_model.pt')
        
        torch.save(model.state_dict(), path)
        self.last_model = path
        