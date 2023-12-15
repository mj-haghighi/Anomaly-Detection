import torch
from enums import LR_SCHEDULER

def apply_lrscheduler(optimizer, scheduler: str):
    if scheduler == LR_SCHEDULER.cosine_annealingLR:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    if scheduler == LR_SCHEDULER.none:
        return None