import os
import os.path as osp
import pandas as pd
from typing import List, Dict
from logger.logger_interface import ILogger
from train.dynamics import Dynamics
from metric.metric_interface import IMetric


class DataframeLogger(ILogger):
    def __init__(self, logdir, base_name="log.pd", metric_columns=None, model_name: str = None, opt_name: str = None) -> None:
        self.logdir = logdir
        self.model_name = model_name
        self.opt_name = opt_name
        self.base_name = base_name
        self.path = osp.join(logdir, base_name)
        self.column_names = ['model', 'optimizer', 'epoch', 'fold',
                             'sample', 'label', 'true_label', 'phase']
        if metric_columns:
            self.column_names.extend(metric_columns)

        self.dataframe: pd.DataFrame = pd.DataFrame(columns=self.column_names)
        if not osp.isdir(logdir):
            os.makedirs(logdir)

    def log(self, fold: int, epoch: int, iteration: int, samples: List[str], phase: str, labels: List[str], true_labels: List[str], metrics: List[IMetric]):
        batch_size = len(samples)
        for i in range(batch_size):
            self.__log_sample(
                fold=fold,
                epoch=epoch, sample=samples[i], phase=phase,
                label=labels[i], true_label=true_labels[i],
                metrics={metric.name: metric.value[i] for metric in metrics}
            )
        self.dataframe.to_pickle(osp.join(self.logdir, self.base_name))

    def __log_sample(self, fold: int, epoch: int, sample: str, phase: str, label: str, true_label: str, metrics: Dict[str, float]):
        data = {
            "fold": fold,
            "model": self.model_name, "optimizer": self.opt_name,
            "epoch": epoch, "sample": sample, "phase": phase,
            "label": label, "true_label": true_label}
        data.update(metrics)

        self.dataframe = self.dataframe._append(data, ignore_index=True)
