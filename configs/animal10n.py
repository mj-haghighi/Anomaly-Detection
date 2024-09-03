from reform import reform_animal10n
from enums import EXT
from configs.ConfigInterface import IConfig


class Config(IConfig):
    download_link = None
    filetype = EXT.ZIP
    datatype = EXT.JPG
    raw_data_folder = 'animal10n'
    classes = ['cat', 'lynx', 'wolf', 'coyote',
               'cheetah', 'jaguer', 'chimpanzee', 'orangutan', 'hamster', 'guinea pig']
    labels = list(range(10))
    mean = [0.5123, 0.4964, 0.4091]
    std = [0.2654, 0.2599, 0.2782]
    trainset = 'train/'
    train_size = 50000
    testset = 'test/'
    validationset = 'validation/'
    reform = reform_animal10n
