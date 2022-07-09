import zipfile
import tarfile
from enums import EXT
import tarfile



def __extract_zip(path, to):
    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(to)

def __extract_targz(path, to):
    tar = tarfile.open(path, "r:gz")
    tar.extractall(path=to)
    tar.close()

def __extract_tarxz(path, to):
    tar = tarfile.open(path, "r:xz")
    tar.extractall(path=to)
    tar.close()

def extract(path, to, type:str):
    if type == EXT.zip:
        __extract_zip(path, to)
    elif type == EXT.targz:
        __extract_targz(path, to)
    elif type == EXT.tarxz:
        __extract_tarxz(path, to)
    else:
        raise Exception("File type: {} not support".format(type))
