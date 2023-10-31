from enums import EXT
from configs.ConfigInterface import IConfig


class Config(IConfig):
    download_link = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filetype = EXT.targz
    datatype = 'todo'
    classes = 'todo'
    mean = 'todo'
    std = 'todo'
    trainset = 'todo'
    validationset = 'todo'
    testset = 'todo'
