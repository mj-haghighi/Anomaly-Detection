import os
import shutil
import pandas as pd

DATASET_NAME = 'animal10n'
BASE_PATH = os.path.join('/home/vision/Repo/cleanset/dataset', DATASET_NAME)
RELATIVE_PATH = os.path.join('dataset', DATASET_NAME)
TRAINSET = 'train'
TESTSET = 'test'
CLASSES = ['cat', 'lynx', 'wolf', 'coyote',
           'cheetah', 'jaguer', 'chimpanzee', 'orangutan', 'hamster', 'guinea pig']


def reform_datset():
    train_images = os.listdir(os.path.join(BASE_PATH, TRAINSET))
    test_images = os.listdir(os.path.join(BASE_PATH, TESTSET))
    info_df = pd.DataFrame(columns=['index', 'path', 'phase', 'true_label'])
    index = 0

    for train_image in train_images:
        class_name = CLASSES[int(train_image.split('_')[0])]
        if not os.path.exists(os.path.join(BASE_PATH, TRAINSET, class_name)):
            os.makedirs(os.path.join(BASE_PATH, TRAINSET, class_name))
        shutil.move(os.path.join(BASE_PATH, TRAINSET, train_image),
               os.path.join(BASE_PATH, TRAINSET, class_name, train_image))
        info_df.loc[len(info_df)] = [index,
                                     str(os.path.join(RELATIVE_PATH, TRAINSET, class_name, train_image)),
                                     'train',
                                     int(train_image.split('_')[0])]
        index += 1

    for test_image in test_images:
        class_name = CLASSES[int(test_image.split('_')[0])]
        if not os.path.exists(os.path.join(BASE_PATH, TESTSET, class_name)):
            os.makedirs(os.path.join(BASE_PATH, TESTSET, class_name))
        shutil.move(os.path.join(BASE_PATH, TESTSET, test_image),
               os.path.join(BASE_PATH, TESTSET, class_name, test_image))
        info_df.loc[len(info_df)] = [index,
                                     str(os.path.join(RELATIVE_PATH, TESTSET, class_name, test_image)),
                                     'test',
                                     int(test_image.split('_')[0])]
        index += 1

    info_df.to_csv(str(os.path.join(BASE_PATH, 'info.csv')), index=False)

if __name__ == "__main__":    
    reform_datset()