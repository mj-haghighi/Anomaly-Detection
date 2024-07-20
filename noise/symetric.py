import numpy as np
import pandas as pd
import os.path as osp
import cleanlab.benchmarking.noise_generation as ng

import sys
sys.path.append('.')

from typing import Tuple
from configs import configs


NOISE_PERSENTAGE_OPTIONS = [0.0, 0.03, 0.07, 0.13]
NOISE_SPARSITY_OPTIONS = [0.0, 0.2, 0.4, 0.6] 


def generate_symmetric_noise(percentage: float, sparsity: float, dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate symmetric noise for a given dataset.

    Args:
        percentage (float): The percentage of noise to introduce, as a fraction of the number of classes.
        sparsity (float): The fraction of zero noise rates.
        dataset_name (str): The name of the dataset.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the noisy labels and the noise matrix.

    Raises:
        Exception: If the dataset name is not recognized or if the dataset info file does not exist.
    """
    if dataset_name not in configs.keys():
        raise Exception(f"Unknown dataset '{dataset_name}'")

    config = configs[dataset_name]
    num_classes = len(config.classes)
    noise_level = percentage * num_classes
    class_probabilities = np.ones(shape=(num_classes)) / num_classes

    noise_matrix = ng.generate_noise_matrix_from_trace(
        py=class_probabilities, K=num_classes,
        trace=num_classes - noise_level,
        frac_zero_noise_rates=sparsity, seed=43
    )

    dataset_info_path = osp.join(config.outdir, dataset_name, 'info.csv')
    if not osp.exists(dataset_info_path):
        raise Exception("Dataset info file does not exist. Use download_dataset to download and create the dataset info file.")

    df = pd.read_csv(dataset_info_path, index_col='index')
    train_df = df[df['phase'] == 'train']
    true_labels = train_df['true_label']
    noisy_labels = ng.generate_noisy_labels(true_labels=true_labels, noise_matrix=noise_matrix)

    return np.array(noisy_labels), np.array(noise_matrix)

