from .download_dataset import download_dataset
from .extract import extract
from enums import LOSS_REDUCTION
from torch import mean
from .loss_reduction import median

loss_reductions = {
    LOSS_REDUCTION.mean: mean,
    LOSS_REDUCTION.median: median
}
