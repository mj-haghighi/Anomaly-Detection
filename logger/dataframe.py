import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import os.path as osp
import pandas as pd
from queue import Queue
from typing import List, Dict
from logger.logger_interface import ILogger
from train.dynamics import Dynamics
from metric.metric_interface import IMetric


class DataframeLogger(ILogger):
    def __init__(self, logdir, base_name="log.pd", logQ: Queue = None, metric_columns=None, model_name: str = None, opt_name: str = None) -> None:
        self.logQ = logQ
        self.logdir = logdir
        self.model_name = model_name
        self.opt_name = opt_name
        self.base_name = base_name
        self.path = osp.join(logdir, base_name)
        self.column_names = ['model', 'optimizer', 'epoch', 'fold', 'iteration', 
                             'sample', 'label', 'phase', 'loss_reduction']
        if metric_columns:
            self.column_names.extend(metric_columns)

        if not osp.isdir(logdir):
            os.makedirs(logdir)

    def log(self, fold: int, epoch: int, iteration: int, samples: List[str], phase: str, loss_reduction:str, labels: List[str], metrics: List[IMetric]):
        self.dataframe: pd.DataFrame = pd.DataFrame(columns=self.column_names)
        batch_size = len(samples)
        for i in range(batch_size):
            self.__log_sample(
                fold=fold, iteration=iteration,
                epoch=epoch, sample=samples[i], phase=phase,
                loss_reduction=loss_reduction,
                label=labels[i],
                metrics={name: value[i] for name, value in metrics}
            )
        self.dataframe.to_pickle(osp.join(self.logdir, f"{fold}|{epoch :03d}|{iteration :04d}|{self.base_name}"))

    def __log_sample(self, fold: int, epoch: int, iteration: int, sample: str, phase: str, loss_reduction:str, label: str, metrics: Dict[str, float]):
        data = {
            "fold": fold, "iteration": iteration,
            "model": self.model_name, "optimizer": self.opt_name,
            "epoch": epoch, "sample": sample, "phase": phase,
            "loss_reduction": loss_reduction,"label": label}
        data.update(metrics)
        self.dataframe = self.dataframe._append(data, ignore_index=True)
