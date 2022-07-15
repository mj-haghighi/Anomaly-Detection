from typing import List
from train.dynamics import Dynamics
from metric.metric_interface import IMetric


class ILogger:
    def log(self, dynamics: Dynamics, metrics: List[IMetric]):
        raise Exception("This methoud is not implemented")