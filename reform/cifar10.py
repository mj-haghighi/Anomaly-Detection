import os
import glob
import pickle
import shutil
import random
import numpy as np
import pandas as pd
import os.path as osp
from PIL import Image
from typing import List

random.seed(47)

def write_images(
    base_dir,
    data_phase,
    batch_labels,
    batch_data,
    file_names,
    count, label_names, 
    database_info
):
    base_dir = osp.join(base_dir, data_phase)
    labels = []
    for i in range(count):
        file_name = file_names[i]
        data = batch_data[i].reshape(3, 32, 32)
        data = np.moveaxis(data, 0, -1)
        label = batch_labels[i]

        img = Image.fromarray(data)
        img_path = osp.join(base_dir, label_names[label], file_name.decode("utf-8"))
        img.save(img_path, format="PNG")
        data_entry = {'path': img_path, 'phase': data_phase, 'true_label': label}
        database_info = database_info._append(data_entry, ignore_index = True)
    return database_info


def reform_datset(
        reform_dir: str,
        data_dir: str
):
    columns = ['path', 'phase', 'true_label']
    # Create an empty DataFrame
    df = pd.DataFrame(columns=columns)

    data_batches_path = osp.join(data_dir, 'data_batch_*')
    test_batch_path = osp.join(data_dir, 'test_batch')
    meta_path = osp.join(data_dir, 'batches.meta')

    file = open(meta_path, mode='rb')
    content = pickle.load(file)
    file.close()

    label_names = content['label_names']
    num_cases_per_batch = content['num_cases_per_batch']

    test = "test"
    train = "train"

    for label in label_names:
        if not osp.isdir(osp.join(reform_dir, train, label)):
            os.makedirs(osp.join(reform_dir, train, label))

        if not osp.isdir(osp.join(reform_dir, test, label)):
            os.makedirs(osp.join(reform_dir, test, label))

    data_files = sorted(glob.glob(data_batches_path))

    for file_path in data_files:
        file = open(file_path, mode='rb')
        content = pickle.load(file, encoding='bytes')
        file.close()

        batch_labels = content[b'labels']
        batch_data = content[b'data']
        file_names = content[b'filenames']

        df = write_images(base_dir=reform_dir,
                     data_phase = train,
                     batch_labels=batch_labels,
                     batch_data=batch_data,
                     file_names=file_names,
                     count=num_cases_per_batch,
                     label_names=label_names,
                     database_info=df)

    file = open(test_batch_path, mode='rb')
    content = pickle.load(file, encoding='bytes')
    file.close()

    batch_labels = content[b'labels']
    batch_data = content[b'data']
    file_names = content[b'filenames']

    df = write_images(base_dir=reform_dir,
                 data_phase = test,
                 batch_labels=batch_labels,
                 batch_data=batch_data,
                 file_names=file_names,
                 count=num_cases_per_batch,
                 label_names=label_names, 
                 database_info=df)
    df.index.name = 'index'
    df.to_csv(osp.join(reform_dir, 'info.csv'))
