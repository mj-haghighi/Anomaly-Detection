from enums import EXT
from configs.ConfigInterface import IConfig

class Config(IConfig):
    download_link = "https://github.com/mj-haghighi/mnist_png/raw/master/mnist.zip"
    filetype = EXT.zip
    datatype = EXT.png
    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    trainset='training/'
    validationset='testing/'
    testset='testing/'