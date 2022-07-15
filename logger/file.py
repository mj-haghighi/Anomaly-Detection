import os
import os.path as osp
from typing import List
from logger.logger_interface import ILogger
from train.dynamics import Dynamics
from metric.metric_interface import IMetric

class FileLogger(ILogger):    
    def __init__(self, logdir, base_name="log.txt") -> None:
        self.logdir = logdir
        self.base_name  = base_name
        self.path = osp.join(logdir, base_name)
        
        if not osp.isdir(logdir):
            os.makedirs(logdir)

    def log(self, t_dynamics: Dynamics, v_dynamics: Dynamics, t_metrics: List[IMetric], v_metrics: List[IMetric]):
        line = super().log(t_dynamics, v_dynamics, t_metrics, v_metrics)
        self.__write_line(line)

    def __write_line(self, line: str):
        with open(self.path, 'a') as f:
            f.write(line + "\n")
