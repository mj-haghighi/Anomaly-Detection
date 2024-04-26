from .equal_to_np import EqualToNP
from .equal_to_np import POILICY_NAME as EqualToNP_POILICY_NAME


def get_data_filtering_policy(
    policy_name,
    metric_name,
    experiment_base_dir,
    experiment_number,
    experiments_info_path,
    experiments_dataset_columns):
    if policy_name == EqualToNP_POILICY_NAME:
        policy = EqualToNP 
    else:
        raise Exception(f"Invalid data filtering policy name {policy_name}")


    return policy(
        metric_name, experiment_base_dir,
        experiment_number, experiments_info_path,
        experiments_dataset_columns
    )


        # ["dataset", "model", "optim", "init", "lr_scheduler", "np", "ns", "lr"]