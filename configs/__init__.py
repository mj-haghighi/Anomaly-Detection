from typing import Dict
from configs.ConfigInterface import IConfig
from configs.mnist import Config as mnist_config
from configs.cifar10 import Config as cifar10_config
from configs.cifar100 import Config as cifar100_config
from enums import DATASETS

configs: Dict[str, IConfig] = {
    DATASETS.mnist: mnist_config,
    DATASETS.cifar10: cifar10_config,
    DATASETS.cifar100: cifar100_config,
}
