
import argparse
import os.path as osp
import torch.nn as nn
import optimizer as optim
from queue import Queue
from enums import PHASE
from enums import PARAMS
from model import models
from saver import best_model
from utils import download_dataset
from configs import configs as dataset_configs
from datetime import datetime
from data.set import GeneralDataset
from threading import Thread
from data.loader import collate_fns
from lrscheduler import apply_lrscheduler
from utils.params import init_kaiming_normal
from metric.level1 import Loss, Proba
from train.trainer import Trainer
from data.transforms import transforms
from logger.dataframe import DataframeLogger
from torch.utils.data import DataLoader
from utils.log_configs import log_configs
from utils.inject_noise_to_dataset import NOISE_PERSENTAGE_OPTIONS, NOISE_SPARSITY_OPTIONS

def parse_args():
    parser = argparse.ArgumentParser(description='start training on dataet')
    parser.add_argument('--dataset',            type=str,   default=None,               choices=['mnist', 'cifar10', 'cifar100'], help='choose dataset')
    parser.add_argument('--model',              type=str,   default='resnet18',         choices=['resnet18', 'resnet34'], help='choose model')
    parser.add_argument('--lr_scheduler',       type=str,   default='none',             choices=['none', 'cosine_annealingLR'], help='choose learning rate scheduler')
    parser.add_argument('--params',             type=str,   default=PARAMS.pretrain,    choices=[PARAMS.pretrain, PARAMS.kaiming_normal], help='choose params initialization')
    parser.add_argument('--epochs',             type=int,   default=100,        help='max number of epochs')
    parser.add_argument('--batch_size',         type=int,   default=64,         help='batch size')
    parser.add_argument('--folds',              type=int,   default=3,          help='number of folds in cross validation')
    parser.add_argument('--lr',                 type=float, default=0.0001,     help='learning rate')
    parser.add_argument('--logdir',             type=str,   default='logs/',    help='log directory')
    parser.add_argument('--device',             type=str,   default='cpu',              choices=['cpu', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3', 'cuda:4', 'cuda:5', 'cuda:6'], help='learning device')
    parser.add_argument('--optimizer',          type=str,   default='sgd',              choices=['adam', 'sgd', 'rmsprobe', 'sparseadam'], help='choose model optimizer')
    parser.add_argument('--noise_percentage',   type=float, default=0.0,                choices=NOISE_PERSENTAGE_OPTIONS, help='injected noise precentage of dataset')
    parser.add_argument('--noise_sparsity',     type=float, default=0.0,                choices=NOISE_SPARSITY_OPTIONS, help='sparsity of injected noise to the dataset (fraction of off-diagonal zeros in noise matrix)')

    args = parser.parse_args()
    return args


def main(argv=None):
    args = parse_args()
    logdir = osp.join(args.logdir, args.dataset, args.model, args.optimizer, args.params, args.lr_scheduler, f'np={args.noise_percentage}', f'ns={args.noise_sparsity}', f'lr={args.lr}')
    log_configs(args, logdir)

    if args.noise_percentage > 0.0001:
        label_column=f"noisy_label[np={args.noise_percentage},ns={args.noise_sparsity}]"
    else:
        label_column="true_label"

    train_taransfm, v_transfm = transforms[args.dataset]
    train_dataset = GeneralDataset(dataset_name=args.dataset, label_column=label_column, phase=PHASE.train, transform=train_taransfm)

    num_classes = len(dataset_configs[args.dataset].classes)
    if args.params == PARAMS.kaiming_normal:
        model = models[args.model](num_classes=num_classes, pretrain=False)
        init_kaiming_normal(model)
    else:
        model = models[args.model](num_classes=num_classes, pretrain=True)

    optimizer = optim.load(name=args.optimizer, model=model, learning_rate=args.lr)
    lr_scheduler = apply_lrscheduler(optimizer, args.lr_scheduler)
    error = nn.CrossEntropyLoss(reduction='none')


    savers = [best_model.MINMetricValueModelSaver(model, savedir=logdir)]

    logQ = Queue()
    level1_metrics = [Loss(), Proba()]
    logger = DataframeLogger(
        logdir=logdir, base_name=f"log.pd",
        metric_columns=[metric.name for metric in level1_metrics],
        model_name=args.model, opt_name=args.optimizer, logQ=logQ
    )

    trainer = Trainer(
        model=model,
        error=error,
        device=args.device,
        batch_size=args.batch_size,
        collate_fn=collate_fns[args.dataset],
        train_dataset=train_dataset,
        num_folds=args.folds,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        num_epochs=args.epochs,
        t_metrics=level1_metrics,
        v_metrics=level1_metrics,
        savers=savers,
        logQ=logQ
    )

    training_thread = Thread(target=trainer.start)
    log_thread = Thread(target=logger.start)
    
    training_thread.start()
    log_thread.start()

    training_thread.join()
    log_thread.join()

if __name__ == "__main__":
    main()
