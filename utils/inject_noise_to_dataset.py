import os
import glob
import random
import os.path as osp
import shutil
from typing import List
from configs import configs

def inject_noise_to_dataset(noise_percentage, dataset_name: str, outdir=None):
    if dataset_name not in configs.keys():
        raise Exception("Unknown dataset '{}'".format(dataset_name))
    

    config = configs[dataset_name]
    original_dataset_dir = osp.join(config.outdir, dataset_name, config.trainset)
    noisy_dataset_dir = osp.join(config.outdir, dataset_name, 'noisy{}-'.format(noise_percentage) + config.trainset)
    if osp.isdir(noisy_dataset_dir):
        print("Noisy dataset already exist in {}".format(osp.join(noisy_dataset_dir)))
        return

    data = []
    noisy_data = []
    for cls in config.classes:
        paths = glob.glob(osp.join(original_dataset_dir, cls ,"*."+config.datatype))
        for path in paths:
            data.append((osp.basename(path), cls))

    num_samples = int((noise_percentage / 100.0) * len(data))
    
    noisy_candidate = random.sample(data, num_samples)
    noisy_candidate_names = [x[0] for x in noisy_candidate]
    counter = 0
    for name, cls in data:
        new_cls = cls
        new_name = name
        if name in noisy_candidate_names:
            cls_options: List[str] = config.classes.copy()
            cls_options.remove(cls)
            new_cls = random.sample(cls_options, 1)[0]
            new_name = 'wrong_{}_'.format(cls) + name
            counter +=1
        noisy_data.append((name, cls, new_name, new_cls))
    for cls in config.classes:
        os.makedirs(osp.join(noisy_dataset_dir, cls))

    for old_name, old_cls, new_name, new_cls in noisy_data:
        old_path = osp.join(original_dataset_dir, old_cls, old_name)
        new_path = osp.join(noisy_dataset_dir, new_cls, new_name)
        shutil.copyfile(old_path, new_path)

    config.trainset = 'noisy{}-'.format(noise_percentage) + config.trainset
    print('{} noisey sample injected!'.format(num_samples))
    print('use {} instead as train set'.format(config.trainset))
    