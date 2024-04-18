import time
from typing import List
from metric.metric_interface import IMetric
from utils.verbose import verbose
from enums import VERBOSE

class ILogger:
    def log(self, fold: int, epoch: int, iteration: int, samples: List[str], phase: str, labels: List[str], metrics: List[IMetric]):
        raise Exception("This is not implemented")

    def start(self, logQ):
        while True:
            log_item = logQ.get()
            verbose(f"LOG-Q has {logQ.qsize()} items", VERBOSE.LEVEL_3)
            if log_item == None or log_item == "EOF":
                return
            self.log(**log_item)
