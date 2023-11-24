
import argparse
import os.path as osp
import torch.nn as nn
import optimizer as optim
from enums import PHASE
from model import models
from saver import best_model
from data.set import datasets
from train.trainer import Trainer
from utils import download_dataset
from configs import configs as dataset_configs
from metric import Acc, Cartography
from data.loader import collate_fns
from data.transforms import transforms
from torch.utils.data import DataLoader
from utils.log_configs import log_configs

from utils.inject_noise_to_dataset import inject_noise_to_dataset
from logger import ConsoleLogger, FileLogger

def parse_args():
    parser = argparse.ArgumentParser(description='start training on dataet')
    parser.add_argument('--dataset', type=str, choices=['mnist', 'cifar10', 'cifar100'], help='choose dataset')
    parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'resnet34'], help='choose model')
    parser.add_argument('--epochs', type=int, default=20, help='max number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--logdir', type=str, default='logs/', help='log directory')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3', 'cuda:4', 'cuda:5', 'cuda:6'], help='learning device')
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['adam', 'sgd', 'rmsprobe', 'sparseadam'], help='choose model optimizer')
    parser.add_argument('--inject_noise', type=float, default=0, choices=[0, 0.03, 0.07, 0.13], help='injected noise precentage of dataset')
    parser.add_argument('--noise_sparsity', type=float, default=0, choices=[0, 0.2, 0.4, 0.6], help='sparsity of injected noise to the dataset (fraction of off-diagonal zeros in noise matrix)')

    args = parser.parse_args()
    return args

def main(argv=None):
    args = parse_args()
    logdir = osp.join(args.logdir, args.dataset, args.model, args.optimizer)
    log_configs(args, logdir)

    download_dataset(args.dataset)
    if args.inject_noise > 0:
        inject_noise_to_dataset(noise_percentage=args.inject_noise, sparsity=args.noise_sparsity, dataset_name=args.dataset)

    t_taransfm, v_transfm = transforms[args.dataset]
    t_dataset = datasets[args.dataset](phase=PHASE.train, transform=t_taransfm)
    v_dataset = datasets[args.dataset](phase=PHASE.validation, transform=v_transfm)

    model = models[args.model](num_classes=len(dataset_configs[args.dataset].classes))
    optimizer = optim.load(name=args.optimizer, model=model, learning_rate=args.lr)
    error = nn.CrossEntropyLoss()
    cartography = Cartography()
    t_metrics = [Acc(), cartography]
    v_metrics = [Acc()]
    loggers = [ConsoleLogger(), FileLogger(logdir)]
    savers = [best_model.MINMetricValueModelSaver(model, savedir=logdir)]
    
    t_loader =  DataLoader(
        dataset=t_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fns[args.dataset]
    )
    
    v_loader =  DataLoader(
        dataset=v_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fns[args.dataset]
    )

    trainer = Trainer(
        model=model,
        error=error,
        device=args.device,
        t_loader=t_loader,
        v_loader=v_loader,
        optimizer=optimizer,
        num_epochs=args.epochs,
        t_metrics=t_metrics,
        v_metrics=v_metrics,
        loggers=loggers,
        savers=savers
    )

    trainer.start()
    cartography.value.to_pickle(osp.join(logdir, "cartography.pkl"))


if __name__ == "__main__":
    main()
