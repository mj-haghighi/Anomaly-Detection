from torchvision import transforms as tv_transforms
from configs.cifar10 import Config


default_train_transform = tv_transforms.Compose([
    tv_transforms.ToTensor(),
    tv_transforms.Normalize(mean=Config.mean, std=Config.std),
])

default_validation_transform = tv_transforms.Compose([
    tv_transforms.ToTensor(),
    tv_transforms.Normalize(mean=Config.mean, std=Config.std),
])
