import pandas as pd
import os.path as osp
from configs.general import EXPERIMENT_COLS, CLASSIFIER_INFERENCE_DIR
from .abstract_class import FilteringPolicy

POILICY_NAME = 'classifier_pruning' 
class ClassifierPruning(FilteringPolicy):
    def __init__(
        self,
        experiment_number=None,
        experiments_info_path=None,
        metric_name=None,
        experiment_base_dir=None,
        experiments_dataset_columns=EXPERIMENT_COLS):

        assert osp.exists(experiments_info_path), f"Experiments info not exist in path {experiments_info_path}"
        self.experiments = pd.read_csv(experiments_info_path, index_col='index')

        target_experiment = self.experiments.loc[experiment_number]
        dataset_name = target_experiment['dataset']
        valid_dataset_name = dataset_name if len(dataset_name.split('_')) <= 1 else dataset_name.split('_')[0]
        self.clf_result = pd.read_pickle(osp.join(CLASSIFIER_INFERENCE_DIR, f"{valid_dataset_name}.pkl"), compression='xz')

    def filter_samples(self):
        filter_df = self.clf_result[self.clf_result['agg_clf'] == 0]
        return filter_df
