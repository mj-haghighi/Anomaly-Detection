from typing import List
from train.dynamics import Dynamics
from metric.metric_interface import IMetric

class ConsoleLogger:
    def log(self, dynamics: Dynamics, metrics: List[IMetric]):
        line = "E: {} | loss: {:.4} ".format(dynamics.epoch, dynamics.loss) 
        for metric in metrics:
            line += "| {} ".format(metric)
        print(line)