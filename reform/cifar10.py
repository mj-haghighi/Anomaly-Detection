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
    data_category,
    batch_labels,
    batch_data,
    file_names,
    count, label_names, 
    database_info
):
    base_dir = osp.join(base_dir, data_category)
    labels = []
    for i in range(count):
        file_name = file_names[i]
        data = batch_data[i].reshape(3, 32, 32)
        data = np.moveaxis(data, 0, -1)
        label = batch_labels[i]

        img = Image.fromarray(data)
        img_path = osp.join(base_dir, label_names[label], file_name.decode("utf-8"))
        img.save(img_path, format="PNG")
        data_entry = {'path': img_path, 'true_label': label, 'category': data_category}
        database_info = database_info._append(data_entry, ignore_index = True)
    return database_info


def split_validation(train_data_dir: str, validation_data_dir: str, validation_split=0.2):
    class_folders = os.listdir(train_data_dir)

    for class_folder in class_folders:
        validation_class_folder = osp.join(validation_data_dir, class_folder)
        os.makedirs(validation_class_folder, exist_ok=True)

    for class_folder in class_folders:
        class_path = os.path.join(train_data_dir, class_folder)
        validation_class_folder = osp.join(validation_data_dir, class_folder)

        image_files = os.listdir(class_path)
        num_validation_samples = int(len(image_files) * validation_split)
        validation_samples = random.sample(
            image_files, num_validation_samples)

        for image in validation_samples:
            src_path = os.path.join(class_path, image)
            dest_path = os.path.join(validation_class_folder, image)
            shutil.move(src_path, dest_path)


def reform_datset(
        reform_dir: str,
        data_dir: str
):
    columns = ['path', 'true_label', 'category']
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
                     data_category = train,
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
                 data_category = test,
                 batch_labels=batch_labels,
                 batch_data=batch_data,
                 file_names=file_names,
                 count=num_cases_per_batch,
                 label_names=label_names, 
                 database_info=df)
    df.to_csv(osp.join(reform_dir, 'info.csv'), index=False)
