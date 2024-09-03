from .equal_to_np import EqualToNP
from .equal_to_np import POILICY_NAME as EqualToNP_POILICY_NAME
from .classifier_pruning import POILICY_NAME as CLF_PRUNING
from .classifier_pruning import ClassifierPruning

def get_data_filtering_policy(
    policy_name,
    metric_name,
    experiment_base_dir,
    experiment_number,
    experiments_info_path,
    experiments_dataset_columns):
    if policy_name == EqualToNP_POILICY_NAME:
        policy = EqualToNP 
    elif policy_name == CLF_PRUNING:
        policy = ClassifierPruning
    else:
        raise Exception(f"Invalid data filtering policy name {policy_name}")


    return policy(
        experiments_info_path=experiments_info_path,
        experiment_number=experiment_number,
        metric_name=metric_name,
        experiment_base_dir=experiment_base_dir,
        experiments_dataset_columns=experiments_dataset_columns
    )


        # ["dataset", "model", "optim", "init", "lr_scheduler", "np", "ns", "lr"]