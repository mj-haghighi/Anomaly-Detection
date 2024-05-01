from typing import Dict
from .ConfigInterface import IConfig
from .mnist import Config as mnist_config
from .cifar10 import Config as cifar10_config
from .cifar100 import Config as cifar100_config
from enums import DATASETS

configs: Dict[str, IConfig] = {
    DATASETS.MNIST: mnist_config,
    DATASETS.CIFAR10: cifar10_config,
    DATASETS.CIFAR100: cifar100_config,
}
