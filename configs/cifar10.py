import os
import glob
import pickle
import os.path as osp

from reform import reform_cifar10
from enums import EXT
from configs.ConfigInterface import IConfig


class Config(IConfig):
    download_link = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filetype = EXT.targz
    datatype = EXT.png
    raw_data_folder = 'cifar-10-batches-py'
    classes = ['airplane', 'automobile', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    mean = [0.49139968, 0.48215827, 0.44653124]
    std = [0.24703233, 0.24348505, 0.26158768]
    trainset = 'train/'
    testset = 'test/'
    validationset = 'validation/'
    reform = reform_cifar10
