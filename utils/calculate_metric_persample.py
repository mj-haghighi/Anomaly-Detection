
import os
import pandas as pd
from multiprocessing import Pool
from functools import reduce
from typing import Dict, Any, List

from functools import reduce
from metric.level2.loss import Loss
from metric.level2.aum import AreaUnderMargin
from metric.level2.id2m import IntegralDiff2Max
from metric.level2.entropy import Entropy
from sklearn.preprocessing import RobustScaler
from metric.level2.top_proba import TopProba
from metric.level2.correctness import Correctness
from configs.general import EXPERIMENT_INFO_PATH, EXPERIMENT_COLS, PHASES, EXPERIMENT_BASE_DIR

AVAILABLE_METRICS : Dict[str, Any] = {
    "correctness": Correctness,
    "entropy": Entropy,
    "id2m": IntegralDiff2Max,
    "loss": Loss,
    "top_proba": TopProba,
    "aum": AreaUnderMargin
}
MEANSTD_METRIC = ["top_proba", "entropy", "loss"]

# Load experiments
# e = pd.read_csv("/home/vision/Repo/cleanset/logs/experiments.csv", index_col='index')
# to_do_index = list(e[e['done'] == True].index)
# exist_index = [x.split('_')[-1][:-4] for x in os.listdir("/home/vision/Repo/cleanset/metric_per_sample")]
# to_do_index = [x for x in to_do_index if x not in exist_index]

to_do_index = ['9bf756402e8', '846ec41c22e', 'f732ee88030', '89997809c70', '72dcb84c9dc', 'a6839c74b79', '69d24355d7a', '6d03f65165d']
to_do_index = ['d5a01a77f63', '6915b1a7d55']


print(f"Number of todo experiments: {len(to_do_index)}")
x = input("Continue? (y/n) ")
if x != 'y':
    exit()

# EXPERIMENT_BASE_DIR = "/mnt/mirshekarihdd/backupcleanset/logs/basic_experiments"

# Function to process each part of the index
def process_experiment_indices(experiment_indices: List[str]):
    for experiment_index in experiment_indices:
        try:
            metrics_per_sample = {}
            print(f"Working on {f'metrics_per_sample_{experiment_index}.csv'}")
            experiments = pd.read_csv(EXPERIMENT_INFO_PATH, index_col='index')
            target_experiment = experiments.loc[experiment_index]

            for metric_name in AVAILABLE_METRICS.keys():
                MetricClass = AVAILABLE_METRICS[metric_name]
                metric = MetricClass(
                    experiment_dir=os.path.join(EXPERIMENT_BASE_DIR, *[str(target_experiment[col]) for col in EXPERIMENT_COLS]),
                    phases=PHASES,
                    folds=target_experiment['folds'],
                    epochs=target_experiment['epochs'],
                    epoch_skip=2,
                    scaler=RobustScaler(with_centering=True, with_scaling=True),
                    raw_dataset_path=f"/home/vision/Repo/cleanset/dataset/{target_experiment['dataset']}/info.csv"
                )
                metric_calculation_result = metric.calculate_metric_per_phase(scale=True)

                if metric_name in MEANSTD_METRIC:
                    metric_calculation_result_train = metric_calculation_result[metric_calculation_result['phase'] == 'train'].drop(columns=['phase'])
                    metric_calculation_result_val = metric_calculation_result[metric_calculation_result['phase'] == 'validation'].drop(columns=['phase'])

                    metric_calculation_result_train = metric_calculation_result_train.rename(columns={'mean': f'mean-train-{metric_name}', 'std': f'std-train-{metric_name}'})
                    metric_calculation_result_val = metric_calculation_result_val.rename(columns={'mean': f'mean-validation-{metric_name}', 'std': f'std-validation-{metric_name}'})

                    metric_calculation_result = pd.merge(metric_calculation_result_train, metric_calculation_result_val, on=['sample', 'label'])
                else:
                    metric_calculation_result_train = metric_calculation_result[metric_calculation_result['phase'] == 'train'].drop(columns=['phase'])
                    metric_calculation_result_val = metric_calculation_result[metric_calculation_result['phase'] == 'validation'].drop(columns=['phase'])

                    metric_calculation_result_train = metric_calculation_result_train.rename(columns={metric_name: f'train-{metric_name}'})
                    metric_calculation_result_val = metric_calculation_result_val.rename(columns={metric_name: f'validation-{metric_name}'})

                    metric_calculation_result = pd.merge(metric_calculation_result_train, metric_calculation_result_val, on=['sample', 'label'])

                print(metric_calculation_result.head())
                metrics_per_sample[metric_name] = metric_calculation_result

            # Merge all DataFrames
            merge_cols = ['sample', 'label']
            metrics_per_sample = reduce(lambda left, right: pd.merge(left, right, on=merge_cols, how='inner'), metrics_per_sample.values())
            metrics_per_sample.to_pickle(f'metric_per_sample/robust_scaler_metrics_per_sample_{experiment_index}.csv', compression="xz")
            print(f"Done {f'metrics_per_sample_{experiment_index}.csv'}")
        except Exception as e:
            print(e, experiment_index)

num_parts = 20
split_indices = [to_do_index[i::num_parts] for i in range(num_parts)]

# Use multiprocessing to process each part
if __name__ == "__main__":
    with Pool(processes=num_parts) as pool:
        pool.map(process_experiment_indices, split_indices)
