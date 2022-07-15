from typing import Dict
from .mnist import collate_fn as mnist_collate_fn
from enums import DATASETS

collate_fns = {
    DATASETS.mnist: mnist_collate_fn
}