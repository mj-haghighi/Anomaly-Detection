import glob
import pandas as pd
import os.path as osp
from metric.level2.abstract_class import AbstractMeanStdMetricClass
from utils.metrics import find_max_proba

class TopProba(AbstractMeanStdMetricClass):
    def __init__(self, experiment_dir, phases, folds, epochs, epoch_skip=0,raw_dataset_path="", scaler=None):
        super().__init__(experiment_dir, phases, folds, epochs, scaler, epoch_skip,raw_dataset_path)

    @property
    def metric_name(self):
        return "top_proba"

    def calculate_metric_on_epochs(self, fold, phase):
        samples_data = pd.DataFrame()
        for epoch in range(self.epoch_skip, self.epochs):
            epoch = f"{epoch :03d}"
            glob_regex = osp.join(self.experiment_dir, str(fold), str(phase), str(epoch), '*.pd')
            iterations_log = sorted(glob.glob(glob_regex))
            if len(iterations_log) == 0:
                print(f"No itteration logs found in fold {fold} / phase {phase}/ epoch {epoch}")
                continue
            try:
                iterations_log = [pd.read_pickle(file_path, compression="xz") for file_path in iterations_log]
            except Exception as e:
                print(e)
                iterations_log = [pd.read_pickle(file_path) for file_path in iterations_log]
                print("Found file without compression!")

            iterations_log = pd.concat(iterations_log, axis=0, ignore_index=True)
            iterations_log[self.metric_name] = iterations_log['proba'].apply(lambda x: find_max_proba(x))
            iterations_log = iterations_log.drop(columns=['proba', 'loss'])
            samples_data = samples_data._append(iterations_log, ignore_index=True)

        metric = samples_data.groupby(['sample', 'label'])[self.metric_name].agg(['mean', 'std']).reset_index()
        return metric