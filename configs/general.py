import os.path as osp

MODELS                          = ["resnet18", "resnet34", "xception"]
DATASETS                        = ["cifar10", "cifar100", "mnist"]
OPTIMS                          = [('adam', ["0.001", "0.0001"]), ('sgd', ["0.1", "0.01"])]
INITIALIZATIONS                 = ['pretrain', 'kaiming_normal']
LR_SCHEDULERS                   = ['none', 'reduceLR']
NOISE_TYPES                     = ["idn", "sym", "asym"]
NPS                             = ["0.0", "0.1", "0.2", "0.3", "0.4", "0.5"]
NSS                             = ["0.2", "0.6"]
DROPOUTS                        = ["0", "0.3"]
PHASES                          = ['train', 'validation']
DATA_FILTERING_POLICIES         = ['equal_to_np']
DATA_RETRIEVAL_POLICIES         = ['remove', 'retrieve_noisy_labels', 'retrieve_all']
TRANSFORM_LEVELS                = ['default', 'intermediate']
EPOCHS                          = 100
FOLDS                           = 3
DEVICE                          = 'cuda:0'


BASE_DIR = "/home/vision/Repo/cleanset/logs"
EXPERIMENT_BASE_DIR = osp.join(BASE_DIR, "basic_experiments")
METRICS_BASE_DIR = osp.join(BASE_DIR, "metrics")
FILTERING_EXPERIMENT_BASE_DIR = osp.join(BASE_DIR, "filtering_experiments")
EXPERIMENT_COLS = ["dataset", "model", "dropout", "optim", "lr", "lr_scheduler", "init", "transform", "noise_type", "np", "ns", "epochs"]
FILTERING_EXPERIMENT_COLS = ['basic_experiment_index', 'based_on', 'data_filtering_policy', 'data_retrieval_policy']

EXPERIMENT_INFO_PATH = osp.join(BASE_DIR, "experiments.csv")
FILTERING_EXPERIMENT_INFO_PATH = osp.join(BASE_DIR, "filtering_experiments.csv")
