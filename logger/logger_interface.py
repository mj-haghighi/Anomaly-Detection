from typing import List
from train.dynamics import Dynamics
from metric.metric_interface import IMetric


class ILogger:
    def log(self, t_dynamics: Dynamics, v_dynamics: Dynamics, t_metrics: List[IMetric], v_metrics: List[IMetric]):
        raise Exception("This methoud is not implemented")