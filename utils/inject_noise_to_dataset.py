import os
import glob
import random
import numpy as np
import os.path as osp
import shutil
from typing import List
from configs import configs

def generate_n_limited_probs_sum_to_m(n, m, min_prob=0.0, max_prob=1.0):

    probs = np.random.dirichlet(np.ones(n)) * m
    
    min_value = min(probs)
    max_value = max(probs)
    while max_value > max_prob:
        new_min = min_value + (max_value - max_prob)
        # This adjustment prevents the new max from always being max_prob.
        adjustment = (max_prob - new_min) * np.random.rand()
        probs[np.argmin(probs)] = new_min + adjustment
        probs[np.argmax(probs)] = max_prob - adjustment
        min_value = min(probs)
        max_value = max(probs)

    min_value = min(probs)
    max_value = max(probs)
    while min_value < (min_prob):
        new_max = max_value - (min_prob - min_value)
        # This adjustment prevents the new min from always being min_prob.
        adjustment = (new_max - min_prob) * np.random.rand()
        probs[np.argmax(probs)] = new_max - adjustment
        probs[np.argmin(probs)] = min_prob + adjustment
        min_value = min(probs)
        max_value = max(probs)
    
    return probs

def generate_noise_matrix(num_classes, noise_rate, sparsity, diversity):
    trace = num_classes - (noise_rate * num_classes)
    noise_matrix = np.zeros(shape=(num_classes, num_classes))
    diagonals = generate_n_limited_probs_sum_to_m(num_classes,
                                          trace,
                                          min_prob=(1.0 - diversity) * (trace / num_classes),
                                          max_prob=np.clip((1.0 + diversity) * (trace / num_classes), None, 1.0))
    np.fill_diagonal(noise_matrix, diagonals)

    off_diagonals_mask = ~np.eye(num_classes, dtype=bool)
    off_diagonals_indices = np.argwhere(off_diagonals_mask)
    np.random.shuffle(off_diagonals_indices)

    num_off_diagonal_non_zeros = int((num_classes * (num_classes - 1)) * (1 - sparsity))
    off_diagonals_indices = off_diagonals_indices[: num_off_diagonal_non_zeros]

    for i in range(noise_matrix.shape[1]):
        m = 1.0 - noise_matrix[i, i]
        filtered_off_diagonals_indices = off_diagonals_indices[off_diagonals_indices[:, 1] == i]
        if len(filtered_off_diagonals_indices) != 0:
            vector = generate_n_limited_probs_sum_to_m(len(filtered_off_diagonals_indices), m)
            for i, idx in enumerate(filtered_off_diagonals_indices):
                noise_matrix[idx[0], idx[1]] = vector[i]
        else:
            continue
        
    return noise_matrix


def inject_noise_to_dataset(noise_percentage, sparsity, dataset_name: str, outdir=None):
    if dataset_name not in configs.keys():
        raise Exception("Unknown dataset '{}'".format(dataset_name))

    config = configs[dataset_name]
    original_dataset_dir = osp.join(config.outdir, dataset_name, config.trainset)
    noisy_dataset_dir = osp.join(config.outdir, dataset_name, 'noisy{}-sparsity{}-'.format(noise_percentage, sparsity) + config.trainset)
    if osp.isdir(noisy_dataset_dir):
        print("Noisy dataset already exist in {}".format(osp.join(noisy_dataset_dir)))
        return

    data = []
    noisy_data = []
    for cls in config.classes:
        paths = glob.glob(osp.join(original_dataset_dir, cls ,"*."+config.datatype))
        for path in paths:
            data.append((osp.basename(path), cls))

    noise_matrix = generate_noise_matrix(num_classes=len(config.classes), noise_rate=noise_percentage, sparsity=sparsity, diversity=0.2)

    num_noisy_samples = 0
    noisy_data = []

    for i, cls in enumerate(config.classes):
        filtered_data = [(data_path, data_cls) for (data_path, data_cls) in data if data_cls == cls]
        probs = noise_matrix[i]
        
        for j, prob in enumerate(probs):
            if i != j:
                noisy_candidates = random.sample(filtered_data, int(prob * len(filtered_data)))
                num_noisy_samples += len(noisy_candidates)
                
                for candid_name, candid_cls in noisy_candidates:
                    new_cls = config.classes[j]
                    new_name = f'wrong_{candid_cls}_{candid_name}'
                    noisy_data.append((candid_name, candid_cls, new_name, new_cls))
                
                filtered_data = [item for item in filtered_data if item not in noisy_candidates]

        noisy_data.extend([(name, cls, name, cls) for name, cls in filtered_data])

    for cls in config.classes:
        os.makedirs(osp.join(noisy_dataset_dir, cls))

    for old_name, old_cls, new_name, new_cls in noisy_data:
        old_path = osp.join(original_dataset_dir, old_cls, old_name)
        new_path = osp.join(noisy_dataset_dir, new_cls, new_name)
        shutil.copyfile(old_path, new_path)

    config.trainset = 'noisy{}-sparsity{}-'.format(noise_percentage, sparsity) + config.trainset
    print('{} noisy samples injected!'.format(num_noisy_samples))
    print('use {} instead as train set'.format(config.trainset))
    