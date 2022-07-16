import torch
import torch.nn as nn
import os.path as osp
from train.dynamics import Dynamics
from metric.metric_interface import IMetric

class IModelSaver:
    def __init__(self, model: nn.Module, savedir: str, helper_in_compare = None, lock_on: IMetric = None) -> None:
        self.model = model
        self.helper = (lambda x: x) if helper_in_compare is None else helper_in_compare
        self.locked_metric = lock_on
        self.savedir = savedir

    def look_for_save(self, v_dynamics, t_dynamics: Dynamics, metric_value=None):
        raise Exception("This method is not implemented")

    def save_model(self):
        path = osp.join(self.savedir, 'best_model.pt')
        print('best model saved!, according to following metric = {:.4}'.format(self.best_value))
        torch.save(self.model.state_dict(), path)