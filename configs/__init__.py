from typing import Dict
from .ConfigInterface import IConfig
from .mnist import Config as mnist_config
from .cifar10 import Config as cifar10_config
from .cifar100 import Config as cifar100_config
from .animal10n import Config as animal10n_config
from .cifar10nag import Config as cifar10nag_config
from .cifar10nws import Config as cifar10nws_config

from enums import DATASETS

configs: Dict[str, IConfig] = {
    DATASETS.MNIST: mnist_config,
    DATASETS.CIFAR10: cifar10_config,
    DATASETS.CIFAR100: cifar100_config,
    DATASETS.ANIMAL10N: animal10n_config,
    DATASETS.CIFAR10NAG: cifar10nag_config,
    DATASETS.CIFAR10NWS: cifar10nws_config
}
