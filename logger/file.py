import os
import os.path as osp
from typing import List
from train.dynamics import Dynamics
from metric.metric_interface import IMetric

class FileLogger:    
    def __init__(self, logdir, base_name="log.txt") -> None:
        self.logdir = logdir
        self.base_name  = base_name
        self.path = osp.join(logdir, base_name)
        
        if not osp.isdir(logdir):
            os.makedirs(logdir)

    def log(self, dynamics: Dynamics, metrics: List[IMetric]):
        line = "E: {} | loss: {:.4} ".format(dynamics.epoch, dynamics.loss) 
        for metric in metrics:
            line += "| {} ".format(metric)
        self.__write_line(line)

    def __write_line(self, line: str):
        with open(self.path, 'a') as f:
            f.write(line + "\n")
