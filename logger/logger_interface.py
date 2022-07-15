from typing import List
from train.dynamics import Dynamics
from metric.metric_interface import IMetric


class ILogger:
    def log(self, t_dynamics: Dynamics, v_dynamics: Dynamics, t_metrics: List[IMetric], v_metrics: List[IMetric]):
        line = "E: {} | t-loss: {:.4} | v-loss: {:.4} ".format(t_dynamics.epoch, t_dynamics.loss, v_dynamics.loss) 
        for metric in t_metrics:
            line += "| t-{} ".format(str(metric))
        for metric in v_metrics:
            line += "| v-{} ".format(str(metric))
        return line