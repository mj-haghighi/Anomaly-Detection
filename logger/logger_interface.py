from typing import List
from train.dynamics import Dynamics
from metric.metric_interface import IMetric


class ILogger:
    def log(self,fold: int, epoch: int, samples: List[str], phase: str, labels: List[str], true_labels: List[str], metrics: List[IMetric]):
        raise Exception("This is not implemented")
