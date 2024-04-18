import time
import torch
import os.path as osp
import argparse
import torch.nn as nn
from torch import multiprocessing
from multiprocessing import Queue
from metric.level1 import Loss, Proba
from utils.inject_noise_to_dataset import NOISE_PERSENTAGE_OPTIONS, NOISE_SPARSITY_OPTIONS
from train.trainer import train_fold, logger

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

def parse_args():
    parser = argparse.ArgumentParser(description='start training on dataet')
    parser.add_argument('--dataset',            type=str,   default=None,               choices=['mnist', 'cifar10', 'cifar100'], help='choose dataset')
    parser.add_argument('--model',              type=str,   default='resnet18',         choices=['resnet18', 'resnet34', 'xception'], help='choose model')
    parser.add_argument('--lr_scheduler',       type=str,   default='none',             choices=['none', 'cosine_annealingLR', 'reduceLR'], help='choose learning rate scheduler')
    parser.add_argument('--params',             type=str,   default='pretrain',    choices=['pretrain', 'kaiming_normal'], help='choose params initialization')
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

# queue = multiprocessing.Queue()
queue_s = [multiprocessing.Queue() for i in range(3)]


# Use multiprocessing to train each fold in parallel
if __name__ == '__main__':
    args = parse_args()
    logdir = osp.join(
        args.logdir,
        args.dataset,
        args.model,
        args.optimizer,
        args.params,
        args.lr_scheduler,
        f'np={args.noise_percentage}',
        f'ns={args.noise_sparsity}',
        f'lr={args.lr}')

    processes = []
    for fold in range(args.folds):
        process_args = {
            'num_folds': args.folds,
            'num_epochs': args.epochs,
            'fold': fold,
            'model_name': args.model,
            'dataset_name': args.dataset,
            'optimizer_name': args.optimizer,
            'learning_rate': args.lr,
            'lr_scheduler_name': args.lr_scheduler,
            'num_classes': 10,
            'logdir': logdir,
            'pretrain': args.params=='pretrain',
            'queue': queue_s[fold],
            'label_column': f"noisy_label[np={args.noise_percentage},ns={args.noise_sparsity}]" if args.noise_percentage > 0.001 else 'true_label'
        }
        p = multiprocessing.Process(target=train_fold, kwargs=process_args)
        processes.append(p)
        p.start()
    
    log_processes = []
    for fold in range(args.folds):
        log_process_args = {
            'queue': queue_s[fold],
            'logdir': logdir,
            'model_name': args.model ,
            'optimizer_name': args.optimizer
        }
        logp = multiprocessing.Process(target=logger, kwargs=log_process_args)
        log_processes.append(logp)
        logp.start()

    for p in processes:
        p.join()

    for p in log_processes:
        p.join()
    
