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
    mean = 'todo'
    std = 'todo'
    trainset = 'train/'
    testset = 'test/'
    reform = reform_cifar10
