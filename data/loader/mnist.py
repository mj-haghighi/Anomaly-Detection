from typing import List
import torch
import numpy as np

def collate_fn(batch):
    img_names: List[str] = []
    imgs: List[torch.Tensor] = []
    clabels: List[torch.Tensor] = []
    for sample in batch:
        img_name, img, clabel = sample
        img_names.append(img_name)
        imgs.append(img)
        clabels.append(clabel)

    return img_names, torch.stack(imgs, 0), torch.stack(clabels, 0)