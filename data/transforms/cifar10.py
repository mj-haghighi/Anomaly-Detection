from torchvision import transforms as tv_transforms
from configs.cifar10 import Config


t_transforms = tv_transforms.Compose([
    tv_transforms.ToTensor(),
    tv_transforms.Normalize(mean=Config.mean, std=Config.std),
    tv_transforms.RandomRotation(degrees=15),
    tv_transforms.RandomAffine(translate=(10, 10), scale=(0.85, 1.15))
])

v_transforms = tv_transforms.Compose([
    tv_transforms.ToTensor(),
    tv_transforms.Normalize(mean=Config.mean, std=Config.std),
])
