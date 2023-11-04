import os
import glob
import pickle
import numpy as np
import os.path as osp

from PIL import Image


def write_images(
    base_dir,
    batch_labels,
    batch_data,
    file_names,
    count, label_names
):
    labels = []
    for i in range(count):
        file_name = file_names[i]
        data = batch_data[i].reshape(3, 32, 32)
        data = np.moveaxis(data, 0, -1)
        label = batch_labels[i]

        img = Image.fromarray(data)
        img.save(osp.join(
            base_dir, label_names[label], file_name.decode("utf-8")), format="PNG")


def reform_datset(
        reform_dir: str,
        data_dir: str
):

    data_batches_path = osp.join(data_dir, 'data_batch_.*')
    test_batch_path = osp.join(data_dir, 'test_batch')
    meta_path = osp.join(data_dir, 'batches.meta')

    file = open(meta_path, mode='rb')
    content = pickle.load(file)
    file.close()

    label_names = content['label_names']
    num_cases_per_batch = content['num_cases_per_batch']

    train = "train"
    test = "test"

    for label in label_names:
        if not osp.isdir(osp.join(reform_dir, train, label)):
            os.makedirs(osp.join(reform_dir, train, label))

        if not osp.isdir(osp.join(reform_dir, test, label)):
            os.makedirs(osp.join(reform_dir, test, label))

    data_files = glob.glob(data_batches_path)
    for file_path in data_files:
        file = open(file_path, mode='rb')
        content = pickle.load(file, encoding='bytes')
        file.close()

        batch_labels = content[b'labels']
        batch_data = content[b'data']
        file_names = content[b'filenames']

        write_images(batch_labels, batch_data, file_names,
                     count=num_cases_per_batch, base_dir=osp.join(reform_dir, train), label_names=label_names)

    file = open(test_batch_path, mode='rb')
    content = pickle.load(file, encoding='bytes')
    file.close()

    batch_labels = content[b'labels']
    batch_data = content[b'data']
    file_names = content[b'filenames']

    write_images(base_dir=osp.join(reform_dir, test),
                 batch_labels=batch_labels,
                 batch_data=batch_data,
                 file_names=file_names,
                 count=num_cases_per_batch,
                 label_names=label_names)
