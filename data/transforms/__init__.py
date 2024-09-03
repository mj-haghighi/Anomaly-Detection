from enums import DATASETS
from enums import TRANSFORM_LEVEL
from . import mnist
from . import cifar10
from . import animal10n
from . import cifar10nag
from . import cifar10nws

default_transforms={
    DATASETS.MNIST: (mnist.default_train_transform, mnist.default_validation_transform),
    DATASETS.CIFAR10: (cifar10.default_train_transform, cifar10.default_validation_transform),
    DATASETS.ANIMAL10N: (animal10n.default_train_transform, animal10n.default_validation_transform),
    DATASETS.CIFAR10NAG: (cifar10nag.default_train_transform, cifar10nag.default_validation_transform),
    DATASETS.CIFAR10NWS: (cifar10nws.default_train_transform, cifar10nws.default_validation_transform)
}

intermediate_transforms={
    DATASETS.CIFAR10: (cifar10.intermediate_train_transform, cifar10.default_validation_transform),
    DATASETS.ANIMAL10N: (animal10n.intermediate_train_transform, animal10n.default_validation_transform),
    DATASETS.CIFAR10NAG: (cifar10nag.intermediate_train_transform, cifar10nag.default_validation_transform),
    DATASETS.CIFAR10NWS: (cifar10nws.intermediate_train_transform, cifar10nws.default_validation_transform)
}


def get_transforms(dataset_name, transform_level: str = TRANSFORM_LEVEL.DEFAULT):
    if transform_level == TRANSFORM_LEVEL.DEFAULT and dataset_name in default_transforms.keys():
        return default_transforms[dataset_name]
    if transform_level == TRANSFORM_LEVEL.INTERMEDIATE and dataset_name in intermediate_transforms.keys():
        return intermediate_transforms[dataset_name]

    raise Exception(f"Can not retrive transform according to dataset: {dataset_name}, transform level: {transform_level}")
