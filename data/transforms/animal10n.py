from torchvision import transforms as tv_transforms
from configs.animal10n import Config

default_validation_transform = tv_transforms.Compose([
    tv_transforms.ToTensor(),
    tv_transforms.Normalize(mean=Config.mean, std=Config.std),
])

default_train_transform = tv_transforms.Compose([
    tv_transforms.ToTensor(),
    tv_transforms.RandomCrop(64, padding=4),
    tv_transforms.RandomHorizontalFlip(),
    tv_transforms.Normalize(mean=Config.mean, std=Config.std),
])

intermediate_train_transform = tv_transforms.Compose([
    tv_transforms.ToTensor(),
    tv_transforms.RandomCrop(64, padding=10),
    tv_transforms.RandomHorizontalFlip(),
    tv_transforms.RandomRotation(degrees=(30,90)),
    tv_transforms.RandomPerspective(distortion_scale=0.4, p=1),
    tv_transforms.Normalize(mean=Config.mean, std=Config.std),
])
