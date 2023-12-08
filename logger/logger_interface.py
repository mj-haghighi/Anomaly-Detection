import time
from queue import Queue
from typing import List
from train.dynamics import Dynamics
from metric.metric_interface import IMetric


class ILogger:
    def __init__(self, logQ: Queue):
        self.logQ = logQ

    def log(self, fold: int, epoch: int, iteration: int, samples: List[str], phase: str, labels: List[str], metrics: List[IMetric]):
        raise Exception("This is not implemented")

    def start(self):
        while True:
            time.sleep(0.05)
            if not self.logQ.empty():
                log_item = self.logQ.get()
                if log_item == "EOF":
                    break
                self.log(**log_item)
                            
