from reform import reform_cifar10nws
from enums import EXT
from configs.ConfigInterface import IConfig


class Config(IConfig):
    download_link = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filetype = EXT.TARGZ
    datatype = EXT.PNG
    raw_data_folder = 'cifar-10-batches-py'
    classes = ['airplane', 'automobile', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    labels = list(range(10))
    mean = [0.49139968, 0.48215827, 0.44653124]
    std = [0.2023, 0.1994, 0.2010]
    trainset = 'train/'
    train_size = 50000
    testset = 'test/'
    validationset = 'validation/'
    reform = reform_cifar10nws
