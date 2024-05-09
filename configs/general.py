import os.path as osp

MODELS                          = ["resnet18", "resnet34", "xception"]
DATASETS                        = ['cifar10']
OPTIMS                          = [('adam', "0.001"), ('sgd', "0.1")]
INITIALIZATIONS                 = ['pretrain', 'kaiming_normal']
LR_SCHEDULERS                   = ['reduceLR', 'none', 'cosine_annealingLR']
NPS                             = ["0.0", "0.03", "0.07", "0.13"]
NSS                             = ["0.0", "0.2", "0.4", "0.6"]
PHASES                          = ['train', 'validation']
DATA_FILTERING_POLICIES         = ['equal_to_np']
DATA_RETRIEVAL_POLICIES         = ['remove', 'retrieve_noisy_labels', 'retrieve_all']
EPOCHS                          = 15
FOLDS                           = 3
DEVICE                          = 'cuda:0'


BASE_DIR = "/home/vision/Repo/cleanset/logs"
EXPERIMENT_BASE_DIR = osp.join(BASE_DIR, "basic_experiments")
FILTERING_EXPERIMENT_BASE_DIR = osp.join(BASE_DIR, "filtering_experiments")
EXPERIMENT_COLS = ["dataset", "model", "optim", "transform", "init", "lr_scheduler", "np", "ns", "lr" ]
FILTERING_EXPERIMENT_COLS = ['basic_experiment_index', 'based_on', 'data_filtering_policy', 'data_retrieval_policy']

EXPERIMENT_INFO_PATH = osp.join(BASE_DIR, "experiments.csv")
FILTERING_EXPERIMENT_INFO_PATH = osp.join(BASE_DIR, "filtering_experiments.csv")
