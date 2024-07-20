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

from noise import generate_idn_noise, generate_symmetric_noise
from configs import configs


NOISE_PERSENTAGE_OPTIONS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
NOISE_SPARSITY_OPTIONS = [0.0, 0.2, 0.4, 0.6] 


def save_noise(noisy_labels, dataset_info_path, noisy_label_col_name):
    df = pd.read_csv(dataset_info_path, index_col='index')
    test_df = df[df['phase'] == 'test']
    test_labels = test_df['true_label']
    df[noisy_label_col_name] = np.concatenate((noisy_labels, test_labels))
    df.to_csv(dataset_info_path)


def save_cm(noise_matrix, path):
    # path = osp.join(config.outdir, dataset_name, noisy_label_col_name + '.png')        
    fig, ax = plt.subplots()
    im = ax.imshow(noise_matrix, cmap='viridis')
    plt.colorbar(im, ax=ax)
    plt.savefig(path)
    print('noisy samples injected!')




##: To use directly
def parse_args():
    parser = argparse.ArgumentParser(description='download dataset')
    parser.add_argument('--dataset', type=str, choices=['mnist', 'cifar10', 'cifar100'], help='Choose dataset')
    parser.add_argument('--percentage', type=float, choices=NOISE_PERSENTAGE_OPTIONS, help='Injected noise precentage of dataset')
    parser.add_argument('--sparsity', type=float, default=0, choices=NOISE_SPARSITY_OPTIONS, help='Sparsity of injected noise (symetric) to the dataset (fraction of off-diagonal zeros in noise matrix)')
    parser.add_argument('--all-symetric-noise-options', type=bool, default=False, help='Inject noise with all symetric noise options')
    parser.add_argument('--noise-type', default="idn", type=str, choices=['idn', 'sym'], help='noise type, available options: sym, idn')
    args = parser.parse_args()
    return args

feature_size = {
    'cifar10': 3*32*32
}

def main(argv=None):
    args = parse_args()
    path = osp.join('dataset', args.dataset, 'info.csv')
    shutil.copy(path, path+".swp")
    if args.noise_type == 'idn':
        noisy_labels, noise_cm = generate_idn_noise(args.percentage, dataset_name=args.dataset, feature_size=feature_size[args.dataset],norm_std=0.1, seed=47)
        col_name=f"idn[np={args.percentage}]"
        save_noise(noisy_labels, dataset_info_path=osp.join('dataset', args.dataset, 'info.csv'), noisy_label_col_name=col_name)
        save_cm(noise_matrix=noise_cm, path=osp.join('dataset', args.dataset,f'{col_name}.png'))
    elif args.noise_type == 'sym':
        if args.all_symetric_noise_options:
            noise_persentage_options = NOISE_PERSENTAGE_OPTIONS[1:]
            noise_sparsity_options = NOISE_SPARSITY_OPTIONS
        else:
            noise_persentage_options = [args.noise_percentage]
            noise_sparsity_options = [args.noise_sparsity]

        for noise_percentage in noise_persentage_options:
            for noise_sparsity in noise_sparsity_options:
                noisy_labels, noise_cm = generate_symmetric_noise(
                    dataset_name=args.dataset,
                    percentage=noise_percentage,
                    sparsity=noise_sparsity)
                col_name=f"sym[np={args.percentage}, ns={args.percentage}]"
                save_noise(noisy_labels, dataset_info_path=osp.join('dataset', args.dataset, 'info.csv'), noisy_label_col_name=col_name)
                save_cm(noise_matrix=noise_cm, path=osp.join('dataset', args.dataset,f'{col_name}.png'))

if __name__ == "__main__":
    main()