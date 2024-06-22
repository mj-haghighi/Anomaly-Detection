import os
import os.path as osp
import shutil
import argparse
from torch import multiprocessing
from typing import List, Dict, Any

from enums import METRIC_TYPE
from metric.level2.loss import Loss
from metric.level2.entropy import Entropy
from metric.level2.top_proba import TopProba
from metric.level2.aum import AreaUnderMargin
from metric.level2.id2m import IntegralDiff2Max
from metric.level2.correctness import Correctness
from metric.level2.abstract_class import AbstractMetricClass
from utils.metrics import load_examins_auc, load_experiments, filter_out_auc_calculated_experiments
from configs.general import EXPERIMENT_INFO_PATH, METRICS_BASE_DIR, EXPERIMENT_COLS, PHASES, EXPERIMENT_BASE_DIR


AVAILABLE_METRICS : Dict[str, Any] = {
    "correctness": Correctness,
    "entropy": Entropy,
    "id2m": IntegralDiff2Max,
    "loss": Loss,
    "top_proba": TopProba,
    "aum": AreaUnderMargin
}

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)

def parse_args():
    parser = argparse.ArgumentParser(description='start calculate metrics auc on dataet')
    
    parser.add_argument('--metrics',           type=str, default='all', help='metrics seperated with `,`')
    parser.add_argument('--workers',           type=int, default=1, help='indicate max workers that run on your machine as process')
    args = parser.parse_args()
    return args


def calculate_auc_for_metric(metric_name):
    assert metric_name in AVAILABLE_METRICS.keys(), f"invalid metric name {metric_name}"
    
    experiments = load_experiments(EXPERIMENT_INFO_PATH, index_col='index')
    experiments = experiments[(experiments['done'] == True) & (experiments['np'] != "np=0.0")]
    experiments_with_auc = load_examins_auc(osp.join(METRICS_BASE_DIR, f"{metric_name}_auc.csv"), index_col='index')

    if experiments_with_auc is not None and not experiments_with_auc.empty:
        experiments = filter_out_auc_calculated_experiments(experiments, experiments_with_auc)

    # Initialize experiments auc with 0.0k
    # if metric.metric_type == METRIC_TYPE.MeanStd:
    #     for metric_agg in ['mean', 'std']:
    #         for phase in PHASES:
    #             experiments[f"{metric_name}-{metric_agg}-{phase}-auc"] = 0.0
    # elif metric.metric_type == METRIC_TYPE.Cumulative:
    #     for phase in PHASES:
    #         experiments[f"{metric_name}-{phase}-auc"] = 0.0

    MetricClass: AbstractMetricClass = AVAILABLE_METRICS[metric_name]
    done_auc_calculations = 0
    total_auc_calculations = len(experiments)
    for index, row in experiments.iterrows():
        metric = MetricClass(
            experiment_dir=osp.join(EXPERIMENT_BASE_DIR, *[str(row[col]) for col in EXPERIMENT_COLS]),
            phases=PHASES,
            folds=row['folds'],
            epochs=row['epochs'],
            epoch_skip=2,
            raw_dataset_path=f"/home/vision/Repo/cleanset/dataset/{row['dataset']}/info.csv")
        metric_calculation_result = metric.calculate_metric_per_phase(scale=True)
        
        if metric.metric_type == METRIC_TYPE.MeanStd:
            mean_metric_auc = metric.calculate_auc_per_phase(metric_calculation_result, metric_name='mean')
            std_metric_auc = metric.calculate_auc_per_phase(metric_calculation_result, metric_name='std')
            for phase in PHASES:
                experiments.at[index, f"{metric_name}-{'mean'}-{phase}-auc"] = mean_metric_auc[phase]
                experiments.at[index, f"{metric_name}-{'std'}-{phase}-auc"] = std_metric_auc[phase]
        elif metric.metric_type == METRIC_TYPE.Cumulative:
            auc = metric.calculate_auc_per_phase(metric_calculation_result)
            for phase in PHASES:
                experiments.at[index, f"{metric_name}-{phase}-auc"] = auc[phase]
        else:
            raise Exception(f"Bug raise: Unknown metric type in metric {metric_name}")

        done_auc_calculations += 1
        experiments.at[index, f"has_auc"] = True
        print(f"{metric_name}: ({done_auc_calculations}/{total_auc_calculations}) AUC calculation '{index}' done!")


    experiments = experiments[(experiments["has_auc"] == True)]
    path = osp.join(METRICS_BASE_DIR, f"{metric_name}_auc.csv")
    if osp.exists(path):
        shutil.copy(path, path+".swp")
    else:
        os.makedirs(METRICS_BASE_DIR, exist_ok=True)
    if experiments_with_auc is not None and not experiments_with_auc.empty:
        experiments_with_auc = experiments_with_auc._append(experiments.drop(columns=['has_auc']))
        experiments_with_auc.sort_index(inplace=True)
        experiments_with_auc.to_csv(path)
    else:
        experiments.drop(columns=['has_auc'], inplace=True)
        experiments.sort_index(inplace=True)
        experiments.to_csv(osp.join(path))

def main():

    args = parse_args()

    metrics: List = [] 
    if args.metrics == "all":
        metrics = AVAILABLE_METRICS.keys()
    else:
        metrics = args.metrics.split(",")
        valid_metrics = [m for m in metrics if m in AVAILABLE_METRICS.keys()]
        assert len(metrics) == len(valid_metrics), f"all metrics are not valid, available metrics to calculate are: {AVAILABLE_METRICS.keys()}"

    print("selected metrics: ", metrics)

    while metrics:
        to_calculate_metrics = metrics[:args.workers]
        del metrics[:args.workers]

        processes = []
        for metric in to_calculate_metrics:
            args = {
                "metric_name": metric,
            }
            p = multiprocessing.Process(target=calculate_auc_for_metric, kwargs=args)
            processes.append(p)
            p.start()

        for p in processes:
            p.join()


if __name__ == "__main__":
    main()