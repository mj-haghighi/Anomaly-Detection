from enums import LR_SCHEDULER
from torch.optim.lr_scheduler import CosineAnnealingLR as TorchCosineAnnealingLR
from torch.optim.lr_scheduler import ReduceLROnPlateau as TorchReduceLROnPlateau

class CosineAnnealingLR(TorchCosineAnnealingLR):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def take_step(self, metrics, epoch=None):
        super().step()


class ReduceLROnPlateau(TorchReduceLROnPlateau):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def take_step(self, metrics, epoch=None):
        super().step(metrics=metrics)



def get_lrscheduler(optimizer, scheduler: str):
    if scheduler == LR_SCHEDULER.COSINE_ANNEALINGLR:
        return CosineAnnealingLR(optimizer=optimizer, T_max=20)
    if scheduler == LR_SCHEDULER.NONE:
        return None
    if scheduler == LR_SCHEDULER.REDUCELR:
        return ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.25, threshold=1e-7, patience=10)