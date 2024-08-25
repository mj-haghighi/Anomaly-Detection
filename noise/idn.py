import numpy as np
import torch
from math import inf
from scipy import stats
import torch.nn.functional as F
import torch.nn as nn
from data.set.dataset import GeneralDataset
from data.transforms import get_transforms, TRANSFORM_LEVEL
from configs import configs
from typing import Tuple

def generate_idn_noise(noise_rate, dataset_name, feature_size, norm_std, seed) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate instance-dependent noisy labels for a given dataset.

    Parameters:
    noise_rate (float): The rate of noise to be added to the labels.
    dataset_name (str): Name of the dataset (e.g., 'mnist', 'cifar10').
    feature_size (int): The size of the input images (e.g., 3*32*32).
    norm_std (float): Standard deviation for the noise distribution.
    seed (int): Random seed for reproducibility.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the noisy labels and the noise matrix.

    Raises:
        Exception: If the dataset name is not recognized or if the dataset info file does not exist.
    """

    if dataset_name not in configs.keys():
      raise Exception(f"Unknown dataset '{dataset_name}'")

    num_classes = len(configs[dataset_name].classes)
    _, transform = get_transforms(dataset_name, TRANSFORM_LEVEL.DEFAULT)
    dataset = GeneralDataset(dataset_name=dataset_name, label_column='true_label', phase='train', transform=transform)
    true_labels = torch.Tensor(np.array(dataset.samples)[:, 2].astype(int))

    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=2 
    )
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    noise_probabilities = []
    flip_distribution = stats.truncnorm((0 - noise_rate) / norm_std, (1 - noise_rate) / norm_std, loc=noise_rate, scale=norm_std)
    flip_rates = flip_distribution.rvs(true_labels.shape[0])

    true_labels = true_labels.cuda()
    noise_weights = torch.FloatTensor(np.random.randn(num_classes, feature_size, num_classes)).cuda()

    for i, (_, image, label) in enumerate(dataloader):
        label = torch.argmax(label)
        label = int(label)
        image = image.cuda()
        logits = image.view(1, -1).mm(noise_weights[label]).squeeze(0)
        logits[label] = -inf
        probabilities = flip_rates[i] * F.softmax(logits, dim=0)
        probabilities[label] += 1 - flip_rates[i]
        noise_probabilities.append(probabilities)

    noise_probabilities = torch.stack(noise_probabilities, 0).cpu().numpy()
    class_indices = list(range(num_classes))
    noisy_labels = [np.random.choice(class_indices, p=noise_probabilities[i]) for i in range(true_labels.shape[0])]

    confusion_matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for true_label, noisy_label in zip(true_labels, noisy_labels):
        true_label, noisy_label = int(true_label), int(noisy_label)
        confusion_matrix[true_label][noisy_label] += 1
    confusion_matrix = np.array(confusion_matrix).astype(float)
    confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=1)
    return np.array(noisy_labels), np.array(confusion_matrix)
