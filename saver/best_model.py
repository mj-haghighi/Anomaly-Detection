import math
import torch.nn as nn
from metric.metric_interface import IMetric
from .saver_interface import IModelSaver

class MAXMetricValueModelSaver(IModelSaver):
    def __init__(self, savedir:str, helper_in_compare = None, lock_on: IMetric = None) -> None:
        super().__init__(savedir, helper_in_compare, lock_on)
        self.best_value = -math.inf

    def look_for_save(self, metric_value: float, epoch: int, model: nn.Module, fold: int = None):
        if self.locked_metric is not None:
            metric_value = self.helper(self.locked_metric.value)
        if metric_value > self.best_value:
            self.best_value = metric_value
            super().save_model(epoch=epoch, fold=fold, model=model)


class MINMetricValueModelSaver(IModelSaver):
    def __init__(self, savedir:str, helper_in_compare = None, lock_on: IMetric = None) -> None:
        super().__init__(savedir, helper_in_compare, lock_on)
        self.best_value = math.inf

    def look_for_save(self, metric_value: float, epoch: int, model: nn.Module, fold: int = None):
        if self.locked_metric is not None:
            metric_value = self.helper(self.locked_metric.value)
        if metric_value < self.best_value:
            self.best_value = metric_value
            super().save_model(epoch=epoch, fold=fold, model=model)