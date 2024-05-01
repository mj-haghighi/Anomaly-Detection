from .mnist import collate_fn as mnist_collate_fn
from .cifar10 import collate_fn as cifar10_collate_fn
from enums import DATASETS

collate_fns = {
    DATASETS.MNIST: mnist_collate_fn,
    DATASETS.CIFAR10: cifar10_collate_fn
}
