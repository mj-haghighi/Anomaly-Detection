import time
from queue import Queue
from typing import List
from train.dynamics import Dynamics
from metric.metric_interface import IMetric


class ILogger:
    def log(self, fold: int, epoch: int, iteration: int, samples: List[str], phase: str, labels: List[str], metrics: List[IMetric]):
        raise Exception("This is not implemented")

    def start(self, logQ):
        while True:
            log_item = logQ.get()
            if log_item == None or log_item == "EOF":
                return
            self.log(**log_item)
