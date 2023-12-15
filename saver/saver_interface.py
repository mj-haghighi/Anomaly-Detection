import os
import torch
import os.path as osp
import torch.nn as nn
from train.dynamics import Dynamics
from metric.metric_interface import IMetric

class IModelSaver:
    def __init__(self, model: nn.Module, savedir: str, helper_in_compare = None, lock_on: IMetric = None) -> None:
        self.model = model
        self.helper = (lambda x: x) if helper_in_compare is None else helper_in_compare
        self.locked_metric = lock_on
        self.savedir = savedir
        self.last_model = None
    def look_for_save(self, metric_value: float, epoch: int, fold: int = None):
        raise Exception("This method is not implemented")            

    def save_model(self, epoch: int, fold: int = None):
        path = self.savedir
        if fold:
            path = osp.join(path, f'{fold}')
        
        if not osp.exists(path):
            os.makedirs(path)
        path = osp.join(path, f'ep{epoch}-best_model.pt')
        print('best model saved!, according to following metric = {:.4}'.format(self.best_value))
        
        if self.last_model is not None and osp.exists(self.last_model):
            os.remove(self.last_model)
        torch.save(self.model.state_dict(), path)
        self.last_model = path