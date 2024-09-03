import os
import numpy as np
import os.path as osp
import pandas as pd
from typing import Dict, Any

from metric.level2.loss import Loss
from metric.level2.entropy import Entropy
from metric.level2.top_proba import TopProba
from metric.level2.aum import AreaUnderMargin
from metric.level2.id2m import IntegralDiff2Max
from metric.level2.correctness import Correctness
from sklearn.preprocessing import RobustScaler


from configs.general import EXPERIMENT_INFO_PATH, EXPERIMENT_COLS, PHASES, EXPERIMENT_BASE_DIR
from functools import reduce

def remove_suffix(value):
    if value.endswith('_a'):
        return value[:-2]
    if value.endswith('_na'):
        return value[:-3]
    else:
        return value

AVAILABLE_METRICS : Dict[str, Any] = {
    "correctness": Correctness,
    "entropy": Entropy,
    "id2m": IntegralDiff2Max,
    "loss": Loss,
    "top_proba": TopProba,
    "aum": AreaUnderMargin
}
MEANSTD_METRIC = ["top_proba", "entropy", "loss"]

to_do_index = ['man10000001','man20000001','man30000001']

print(f"number of todo experiment: {len(to_do_index)}")
if len(to_do_index) < 10:
    print(to_do_index)

x = input("continue? (y/n)")
if x != 'y':
    exit()

# EXPERIMENT_BASE_DIR = "/mnt/mirshekarihdd/backupcleanset/logs/basic_experiments"

for experiment_index in to_do_index:#  to_do_index: 
    try:
        print(f"Working on {f'metrics_per_sample_{experiment_index}.csv'}")
        experiments = pd.read_csv(EXPERIMENT_INFO_PATH, index_col='index')
        target_experiment = experiments.loc[experiment_index]
        valid_dataset_name = target_experiment['dataset'].split('_')[0] if len(target_experiment['dataset'].split('_')) > 1 else target_experiment['dataset'] 
        metrics_per_sample = {}
        for metric_name in AVAILABLE_METRICS.keys():
            MetricClass = AVAILABLE_METRICS[metric_name]
            metric = MetricClass(
                experiment_dir=osp.join(EXPERIMENT_BASE_DIR, *[str(target_experiment[col]) for col in EXPERIMENT_COLS]),
                phases=PHASES,
                folds=target_experiment['folds'],
                epochs=target_experiment['epochs'],
                epoch_skip=2,
                scaler=RobustScaler(with_centering=True, with_scaling=True),
                raw_dataset_path=f"/home/vision/Repo/cleanset/dataset/{valid_dataset_name}/info.csv")
            metric_calculation_result = metric.calculate_metric_per_phase(scale=True)
            
            if metric_name in MEANSTD_METRIC:
                metric_calculation_result_train_a = metric_calculation_result[(metric_calculation_result['phase'] == 'train') & (metric_calculation_result['sample'].str.contains('_a'))].drop(columns=['phase'])
                metric_calculation_result_train_na = metric_calculation_result[(metric_calculation_result['phase'] == 'train') & (metric_calculation_result['sample'].str.contains('_na'))].drop(columns=['phase'])
                metric_calculation_result_train_a['sample']  = metric_calculation_result_train_a['sample'].apply(remove_suffix)
                metric_calculation_result_train_na['sample'] = metric_calculation_result_train_na['sample'].apply(remove_suffix)
                
                metric_calculation_result_val = metric_calculation_result[metric_calculation_result['phase'] == 'validation'].drop(columns=['phase'])


                metric_calculation_result_train_a = metric_calculation_result_train_a.rename(columns={'mean': f'mean-train-{metric_name}_a', 'std': f'std-train-{metric_name}_a'})
                metric_calculation_result_train_na = metric_calculation_result_train_na.rename(columns={'mean': f'mean-train-{metric_name}_na', 'std': f'std-train-{metric_name}_na'})
                metric_calculation_result_train_a['sample'] = metric_calculation_result_train_a['sample'].astype(int)
                metric_calculation_result_train_na['sample'] = metric_calculation_result_train_na['sample'].astype(int)

                metric_calculation_result_val = metric_calculation_result_val.rename(columns={'mean': f'mean-validation-{metric_name}', 'std': f'std-validation-{metric_name}'})

                metric_calculation_result = pd.merge(metric_calculation_result_train_a, metric_calculation_result_train_na, on=['sample', 'label'])
                metric_calculation_result = pd.merge(metric_calculation_result, metric_calculation_result_val, on=['sample', 'label'])
            else:
                metric_calculation_result_train_a            = metric_calculation_result[(metric_calculation_result['phase'] == 'train') & (metric_calculation_result['sample'].str.contains('_a'))].drop(columns=['phase'])
                metric_calculation_result_train_na           = metric_calculation_result[(metric_calculation_result['phase'] == 'train') & (metric_calculation_result['sample'].str.contains('_na'))].drop(columns=['phase'])
                metric_calculation_result_train_a['sample']  = metric_calculation_result_train_a['sample'].apply(remove_suffix)
                metric_calculation_result_train_na['sample'] = metric_calculation_result_train_na['sample'].apply(remove_suffix)

                metric_calculation_result_val = metric_calculation_result[metric_calculation_result['phase'] == 'validation'].drop(columns=['phase'])

                metric_calculation_result_train_a = metric_calculation_result_train_a.rename(columns={metric_name: f'train-{metric_name}_a'})
                metric_calculation_result_train_na = metric_calculation_result_train_na.rename(columns={metric_name: f'train-{metric_name}_na'})
                metric_calculation_result_train_a['sample'] = metric_calculation_result_train_a['sample'].astype(int) 
                metric_calculation_result_train_na['sample'] = metric_calculation_result_train_na['sample'].astype(int) 
                metric_calculation_result_val = metric_calculation_result_val.rename(columns={metric_name: f'validation-{metric_name}'})

                metric_calculation_result = pd.merge(metric_calculation_result_train_a, metric_calculation_result_train_na, on=['sample', 'label'])
                metric_calculation_result = pd.merge(metric_calculation_result, metric_calculation_result_val, on=['sample', 'label'])

            metrics_per_sample[metric_name] = metric_calculation_result

        # print('metrics_per_sample.head()', metrics_per_sample['correctness'])
        # Merge all DataFrames using functools.reduce and pd.merge
        merge_cols = ['sample', 'label']
        metrics_per_sample = reduce(lambda left, right: pd.merge(left, right, on=merge_cols, how='inner'), metrics_per_sample.values())
        metrics_per_sample.to_pickle(f'metric_per_sample/robust_scaler_metrics_per_sample_{experiment_index}.csv', compression="xz")
        print(f"done {f'metrics_per_sample_{experiment_index}.csv'}")
    except Exception as e:
        print(e)
