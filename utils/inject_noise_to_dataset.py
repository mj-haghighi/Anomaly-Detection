import os
import glob
import random
import shutil
import argparse
import numpy as np
import pandas as pd
import os.path as osp
import matplotlib.pyplot as plt
import cleanlab.benchmarking.noise_generation as ng

from typing import List
import sys
sys.path.append('.')

from configs import configs


NOISE_PERSENTAGE_OPTIONS = [0.03, 0.07, 0.13]
NOISE_SPARSITY_OPTIONS = [0.0, 0.2, 0.4, 0.6] 

def inject_noise_to_dataset(percentage, sparsity, dataset_name: str, outdir=None):
    if dataset_name not in configs.keys():
        raise Exception("Unknown dataset '{}'".format(dataset_name))
    config = configs[dataset_name]
    num_classes = len(config.classes)
    noise_level = percentage * num_classes
    py = np.ones(shape=(num_classes)) / num_classes
    noise_matrix = ng.generate_noise_matrix_from_trace(
                    py=py, K=num_classes,
                    trace=num_classes - noise_level,
                    frac_zero_noise_rates=sparsity, seed=43)
    dataset_info_path = osp.join(config.outdir, dataset_name, 'info.csv') 
    if not osp.exists(dataset_info_path):
        raise Exception("Dataset info file does not exist. use download_dataset to download and create dataset info file")
    df = pd.read_csv(dataset_info_path, index_col='index')
    true_labels = df['true_label']
    noisy_labels = ng.generate_noisy_labels(true_labels=true_labels, noise_matrix=noise_matrix)
    noisy_label_col_name = f"noisy_label[np={percentage},ns={sparsity}]" 
    df[noisy_label_col_name] = noisy_labels
    df.to_csv(dataset_info_path)
    
    path = osp.join(config.outdir, dataset_name, noisy_label_col_name + '.png')        
    fig, ax = plt.subplots()
    im = ax.imshow(noise_matrix, cmap='viridis')
    plt.colorbar(im, ax=ax)
    plt.savefig(path)

    print('noisy samples injected!')
    


##: To use directly
def parse_args():
    parser = argparse.ArgumentParser(description='download dataset')
    parser.add_argument('--dataset', type=str, choices=['mnist', 'cifar10', 'cifar100'], help='choose dataset')
    parser.add_argument('--percentage', type=float, choices=NOISE_PERSENTAGE_OPTIONS, help='injected noise precentage of dataset')
    parser.add_argument('--sparsity', type=float, default=0, choices=NOISE_SPARSITY_OPTIONS, help='sparsity of injected noise to the dataset (fraction of off-diagonal zeros in noise matrix)')
    parser.add_argument('--all-noise-options', type=bool, default=False, help='Inject noise with all noise options')

    args = parser.parse_args()
    return args


def main(argv=None):
    args = parse_args()
    if args.all_noise_options:
        noise_persentage_options = NOISE_PERSENTAGE_OPTIONS
        noise_sparsity_options = NOISE_SPARSITY_OPTIONS
    else:
        noise_persentage_options = [args.noise_percentage]
        noise_sparsity_options = [args.noise_sparsity]

    for noise_percentage in noise_persentage_options:
        for noise_sparsity in noise_sparsity_options:
            inject_noise_to_dataset(
                dataset_name=args.dataset,
                percentage=noise_percentage,
                sparsity=noise_sparsity)

if __name__ == "__main__":    
    main()