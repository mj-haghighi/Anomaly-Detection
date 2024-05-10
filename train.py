import os.path as osp
import pandas as pd
import argparse
from torch import multiprocessing
from train.trainer import train_fold, logger
from configs.general import FILTERING_EXPERIMENT_BASE_DIR, EXPERIMENT_BASE_DIR, EXPERIMENT_COLS, FILTERING_EXPERIMENT_INFO_PATH, EXPERIMENT_INFO_PATH

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

def parse_args():
    parser = argparse.ArgumentParser(description='start training on dataet')
    
    parser.add_argument('--experiment_number',           type=str, help='experiment index in experiments dataset')
    parser.add_argument('--filtering_experiment_number', type=str, help='experiment index in filtering experiments dataset')

    args = parser.parse_args()
    return args

# queue = multiprocessing.Queue()
queue_s = [multiprocessing.Queue(30) for i in range(3)]


# Use multiprocessing to train each fold in parallel
if __name__ == '__main__':
    args = parse_args()

    filtering_policy = None
    based_on = None
    if args.filtering_experiment_number is not None:
        filtering_experiments = pd.read_csv(FILTERING_EXPERIMENT_INFO_PATH, index_col='index')
        target_filtering_experiment = filtering_experiments.loc[args.filtering_experiment_number]
        experiment_number = target_filtering_experiment['basic_experiment_index']
        filtering_policy = target_filtering_experiment['data_filtering_policy']
        retrieval_policy = target_filtering_experiment['data_retrieval_policy']
        based_on = target_filtering_experiment['based_on']
        experiment_dir = osp.join(FILTERING_EXPERIMENT_BASE_DIR, str(experiment_number), based_on, filtering_policy, retrieval_policy) 
        experiments = pd.read_csv(EXPERIMENT_INFO_PATH, index_col='index')
        target_experiment = experiments.loc[experiment_number]
    else:
        experiment_number = args.experiment_number
        experiments = pd.read_csv(EXPERIMENT_INFO_PATH, index_col='index')
        target_experiment = experiments.loc[experiment_number]
        experiment_dir = osp.join(EXPERIMENT_BASE_DIR, *[str(target_experiment[col]) for col in EXPERIMENT_COLS])    

    folds = target_experiment['folds']
    processes = []
    for fold in range(folds):
        process_args = {
            'fold': fold,
            'queue': queue_s[fold],
            'experiment_number': experiment_number,
            'filtering_policy': filtering_policy,
            'based_on': based_on,
            'experiment_dir': experiment_dir
        }
        p = multiprocessing.Process(target=train_fold, kwargs=process_args)
        processes.append(p)
        p.start()

    log_processes = []
    
    for fold in range(folds):
        log_process_args = {
            'queue': queue_s[fold],
            'logdir': experiment_dir
        }
        logp = multiprocessing.Process(target=logger, kwargs=log_process_args)
        log_processes.append(logp)
        logp.start()

    for p in processes:
        p.join()

    for p in log_processes:
        p.join()
