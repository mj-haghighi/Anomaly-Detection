import glob
import torch
import os.path as osp

from PIL import Image
from enums import PHASE
from enums import DATASETS
from configs.mnist import Config
from utils import to_categorical
from .dataset_interface import IDataset
from torch.utils.data import Dataset as TorchDataset

class Dataset(TorchDataset, IDataset):
    def __init__(self, phase:str=PHASE.train, transform=None):
        self.datadir = self.__get_datadir(phase)
        self.samples = self.__collect_samples()
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, img_name = self.samples[idx]
        clabel = to_categorical.sample(label, Config.classes)
        clabel = torch.Tensor(clabel)
        img = Image.open(img_path)
        
        if self.transform:
            img = self.transform(img)
        return img_name, img, clabel

    def __get_datadir(self, phase):
        if phase == PHASE.train:
            return osp.join(Config.outdir, DATASETS.mnist, Config.trainset)
        elif phase == PHASE.validation:
            return osp.join(Config.outdir, DATASETS.mnist, Config.validationset)
        elif phase == PHASE.test:
            return osp.join(Config.outdir, DATASETS.mnist, Config.testset)
        else:
            raise Exception("Unkown phase: {}".format(phase))
    
    def __collect_samples(self):
        samples = []
        for cls in Config.classes:
            new_paths = glob.glob(osp.join(self.datadir, cls ,"*."+Config.datatype))
            samples.extend([(p, cls, osp.basename(p)) for p in new_paths])
        return samples