from typing import List
from logger.logger_interface import ILogger
from train.dynamics import Dynamics
from metric.metric_interface import IMetric

class ConsoleLogger(ILogger):
    
    def log(self, t_dynamics: Dynamics, v_dynamics: Dynamics, t_metrics: List[IMetric], v_metrics: List[IMetric]):
        line = super().log(t_dynamics, v_dynamics, t_metrics, v_metrics)
        print(line)