from abc import ABC, abstractmethod
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from utils.metrics import std_agg

class AbstractMeanStdMetricClass(ABC):
    def __init__(self, experiment_dir, phases, folds, epochs, epoch_skip=0, raw_dataset_path=""):
        self.experiment_dir = experiment_dir
        self.phases         = phases
        self.folds          = folds
        self.epochs         = epochs
        self.epoch_skip     = epoch_skip
        self.dataset_info   = pd.read_csv(raw_dataset_path, index_col='index')
        self.scaler         = MinMaxScaler()


    @property
    @abstractmethod
    def metric_name(self):
        pass

    def calculate_auc_per_phase(self, metric, metric_name = None):
        result = {}
        for phase in self.phases:
            phase_metrics = metric[metric['phase'] == phase]
            merged_df = pd.merge(phase_metrics, self.dataset_info[['true_label']], left_on='sample', right_index=True, how='inner')
            y_true = merged_df['true_label'] == merged_df['label']
            if len(y_true.unique()) <= 1:
                result[phase] = None
            else:
                auc_score = roc_auc_score(y_true=y_true, y_score = merged_df[metric_name if metric_name is not None else self.metric_name])
                result[phase] = auc_score
        return result


    def calculate_metric_per_phase(self, scale=False):
        metric = pd.DataFrame()
        for phase in self.phases:
            res = self.calculate_metric_on_folds(phase)
            res['phase'] = phase
            if scale:
                res['mean'] = self.scaler.fit_transform(res[['mean']])
                res['std'] = self.scaler.fit_transform(res[['std']])
            metric = metric._append(res, ignore_index = True)
        return metric


    def calculate_metric_on_folds(self, phase):
        metric = pd.DataFrame()
        for fold in range(self.folds):
            res = self.calculate_metric_on_epochs(fold, phase)
            metric = metric._append(res, ignore_index = True)
        
        metric_mean = metric.groupby(by=['sample', 'label'])['mean'].agg(['mean']).reset_index()
        metric_std = metric.groupby(by=['sample', 'label'])['std'].agg(std_agg).reset_index()
        metric_mean['std'] = metric_std['std']
        return metric_mean

    @abstractmethod
    def calculate_metric_on_epochs(self, fold, phase):
        """
        Calculate `mean` and `std` of proposed metric per sample and provide `mean`, `std` columns in a resulting dataframe.
        return:
            `pd.Dataframe`: with columns [`sample`, `label`, `mean`, `std`].
                \n- `sample`: sample index in dataset.
                \n- `label`: label of sample in dataset.
                \n- `mean`: mean of calculated metric throw all epochs.
                \n- `std`: std of calculated metric throw all epochs.
        """
        pass
