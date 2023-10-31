from typing import Dict
from configs.ConfigInterface import IConfig
from configs.mnist import Config as mnist_config
from configs.cfar10 import Config as cfar10_config
from configs.cfar100 import Config as cfar100_config
from enums import DATASETS

configs: Dict[str, IConfig] = {
    DATASETS.mnist: mnist_config,
    DATASETS.cfar10: cfar10_config,
    DATASETS.cfar100: cfar100_config,
}
