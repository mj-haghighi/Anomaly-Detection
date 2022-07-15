from torchvision import transforms as tv_transforms
from configs.mnist import Config

t_transforms = tv_transforms.Compose([
        tv_transforms.ToTensor(),
        # tv_transforms.Grayscale(),
        tv_transforms.Normalize(mean=Config.mean, std=Config.std),
        tv_transforms.RandomRotation(degrees=15)
    ])

v_transforms = tv_transforms.Compose([
        tv_transforms.ToTensor(),
        # tv_transforms.Grayscale(),
        tv_transforms.Normalize(mean=Config.mean, std=Config.std),
    ])
    