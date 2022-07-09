from typing import Dict
from configs.ConfigInterface import IConfig
from configs.mnist import Config as mnist_config
from enums import DATASETS

configs: Dict[str, IConfig] = {
    DATASETS.mnist: mnist_config,
}
