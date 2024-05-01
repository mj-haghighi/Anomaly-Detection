class DATASETS:
    MNIST       = "mnist"
    CIFAR10     = "cifar10"
    CIFAR100    = "cifar100"

class TRANSFORM_LEVEL:
    DEFAULT         = "default"
    INTERMEDIATE    = "intermediate"
    HEAVY           = "heavy"

class CARTOGRAPHY_METRICS:
    CONFIDENCE  = "confidence"
    VARIABILITY = "variability"
    CORRECTNESS = "correctness"

class POLICY:
    FILTER_RATE = "eq_np"

class EXT:
    ZIP     = "zip"
    RAR     = "rar"
    JPG     = "jpg"
    PNG     = "png"
    TARGZ   = "tar.gz"
    TARXZ   = "tar.xz"

class VERBOSE:
    LEVEL_3 = 3
    LEVEL_2 = 2
    LEVEL_1 = 1

class LR_SCHEDULER:
    NONE                = "none"
    REDUCELR            = "reduceLR"
    COSINE_ANNEALINGLR  = "cosine_annealingLR"

class MODELS:
    RESNET18 = "resnet18"
    RESNET34 = "resnet34"
    XCEPTION = "xception"

class OPTIMIZER:
    ADAM        = "adam"
    SGD         = "sgd"
    RMSPROBE    = "rmsprobe"
    SPARSEADAM  = "sparseadam"

class PARAMS:
    PRETRAIN        = "pretrain"
    KAIMING_NORMAL  = "kaiming_normal"

class PHASE:
    TRAIN       = "train"
    VALIDATION  = "validation"
    TEST        = "test"