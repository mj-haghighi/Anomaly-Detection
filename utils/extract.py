import zipfile
from enums import EXT

def __extract_zip(path, to):
    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(to)

def extract(path, to, type:str=EXT.zip):
    if type == EXT.zip:
        __extract_zip(path, to)