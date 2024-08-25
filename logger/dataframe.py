import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import numpy as np
import os.path as osp
import pandas as pd
from queue import Queue
from typing import List, Dict
from logger.logger_interface import ILogger
from train.dynamics import Dynamics
from metric.metric_interface import IMetric


class DataframeLogger(ILogger):
    def __init__(self, logdir, base_name="log.pd", metric_columns=None) -> None:
        self.logdir = logdir
        self.base_name = base_name
        self.path = osp.join(logdir, base_name)
        self.column_names = ['sample', 'label']
        if metric_columns:
            self.column_names.extend(metric_columns)

        if not osp.isdir(logdir):
            os.makedirs(logdir, exist_ok=True)

    def log(self, fold: int, epoch: int, iteration: int, samples: List[str], phase: str, labels: List[str], metrics: List[IMetric]):
        self.dataframe: pd.DataFrame = pd.DataFrame(columns=self.column_names)
        batch_size = len(samples)
        for i in range(batch_size):
            self.__log_sample(
                sample=samples[i], label=labels[i],
                metrics={name: value[i] for name, value in metrics}
            )
        directory = osp.join(self.logdir, f"{fold}", phase, f"{epoch :03d}")
        if not osp.exists(directory):
            os.makedirs(directory, exist_ok=True)

        self.dataframe['sample'] = self.dataframe['sample'].astype('int32')
        self.dataframe['label'] = self.dataframe['label'].astype('int8')
        self.dataframe['loss'] = self.dataframe['loss'].astype('float16')
        self.dataframe['proba'] = self.dataframe['proba'].apply(lambda x: x.astype(np.float16))

        self.dataframe.to_pickle(osp.join(directory, f"{iteration :04d}-{self.base_name}"), compression="xz")

    def __log_sample(self, sample: str, label: str, metrics: Dict[str, float]):
        data = {"sample": sample, "label": label}
        data.update(metrics)
        self.dataframe = self.dataframe._append(data, ignore_index=True)
