import os
import pandas as pd
import os.path as osp
from configs.general import EXPERIMENT_COLS, CLASSIFIER_INFERENCE_DIR

from .abstract_class import FilteringPolicy

POILICY_NAME = 'equal_to_np' 
class EqualToNP(FilteringPolicy):
    def __init__(
        self,
        metric_name,
        experiment_base_dir,
        experiment_number,
        experiments_info_path,
        experiments_dataset_columns):
        
        self.metric_name = metric_name
        
        assert osp.exists(experiments_info_path), f"Experiments info not exist in path {experiments_info_path}"
        self.experiments = pd.read_csv(experiments_info_path, index_col='index')
        
        target_experiment = self.experiments.loc[experiment_number]
        self.target_np = float(target_experiment['np'][3:])
        
        experiment_dir = osp.join(experiment_base_dir, *[str(target_experiment[col]) for col in experiments_dataset_columns])
        metric_results_path = osp.join(experiment_dir, f"{metric_name}_per_sample.csv")
        assert osp.exists(metric_results_path), f"No such metric {metric_name} calculated for this experiment {experiment_dir}"
        self.metric_calculation_result = pd.read_csv(metric_results_path, index_col="sample")

    def filter_samples(self):
        self.metric_calculation_result = self.metric_calculation_result.sort_values(self.metric_name)
        number_of_filter_out_samples = int(self.target_np * len(self.metric_calculation_result))
        filter_out_samples = self.metric_calculation_result[:number_of_filter_out_samples]
        return filter_out_samples
