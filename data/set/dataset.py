import glob
import torch
import pandas as pd
import os.path as osp
from PIL import Image
from enums import PHASE
from enums import DATASETS
from utils import to_categorical
from configs import configs
from data.set.dataset_interface import IDataset
from torch.utils.data import Dataset as TorchDataset


class GeneralDataset(TorchDataset, IDataset):
    def __init__(self, dataset_name: str, label_column, phase, transform=None):
        self.phase          = phase
        self.dataset_name   = dataset_name
        self.label_column   = label_column
        self.dataset_config = configs[dataset_name]
        self.transform      = transform
        self.samples        = self.__collect_samples()
        print('{} sample available in {} set'.format(self.dataset_length, phase))

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        index, img_path, label = self.samples[idx]
        clabel = to_categorical.sample(label, self.dataset_config.labels)
        clabel = torch.Tensor(clabel)
        img = Image.open(img_path)

        if self.transform:
            img = self.transform(img)
        return index, img, clabel

    def __collect_samples(self):
        dataset_info_path = osp.join(self.dataset_config.outdir, self.dataset_name, 'info.csv')
        df = pd.read_csv(dataset_info_path, index_col='index')
        df = df[df['phase'] == self.phase]
        if self.label_column in df.columns:
            indices, paths, labels = df.index.values, df['path'], df[self.label_column]
            self.dataset_length = len(df)
            return list(zip(indices, paths, labels))
        else:
            raise(f"label_column: {self.label_column} is not valid")
