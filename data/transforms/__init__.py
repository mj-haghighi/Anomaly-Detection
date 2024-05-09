from enums import DATASETS
from enums import TRANSFORM_LEVEL
from . import mnist
from . import cifar10

default_transforms={
    DATASETS.MNIST: (mnist.default_train_transform, mnist.default_validation_transform),
    DATASETS.CIFAR10: (cifar10.default_train_transform, cifar10.default_validation_transform)
}


def get_transforms(dataset_name, transform_level=TRANSFORM_LEVEL.DEFAULT):
    if transform_level == TRANSFORM_LEVEL.DEFAULT and dataset_name in default_transforms.keys():
        return default_transforms[dataset_name]

    raise Exception(f"Can not retrive transform according to dataset: {dataset_name}, transform level: {transform_level}")
