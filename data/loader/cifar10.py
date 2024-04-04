from typing import List
import torch
import numpy as np


def collate_fn(batch):
    indices: List[int] = []
    imgs: List[torch.Tensor] = []
    clabels: List[torch.Tensor] = []
    for sample in batch:
        index, img, clabel = sample
        indices.append(index)
        imgs.append(img)
        clabels.append(clabel)

    return indices, torch.stack(imgs, 0), torch.stack(clabels, 0)
