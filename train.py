
import argparse
import os.path as osp
import torch.nn as nn
from enums import PHASE
from model import models
from torch.optim import Adam
from saver import best_model
from data.set import datasets
from train.trainer import Trainer
from utils import download_dataset
from metric import Acc, Cartography
from data.loader import collate_fns
from data.transforms import transforms
from torch.utils.data import DataLoader
from utils.log_configs import log_configs
from logger import ConsoleLogger, FileLogger

def parse_args():
    parser = argparse.ArgumentParser(description='start training on dataet')
    parser.add_argument('--task', type=str, choices=['mnist', 'cfar100'], help='choose task')
    parser.add_argument('--epochs', type=int, default=20, help='max number of epochs')
    parser.add_argument('--batch_size', type=int, default=20, help='batch size')
    parser.add_argument('--lr', type=int, default=0.0001, help='learning rate')
    parser.add_argument('--logdir', type=str, default='logs/', help='log directory')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3', 'cuda:4', 'cuda:5', 'cuda:6'], help='learning device')

    args = parser.parse_args()
    return args

def main(argv=None):
    args = parse_args()
    logdir = osp.join(args.logdir, args.task)
    log_configs(args, logdir)

    download_dataset(args.task)
    t_taransfm, v_transfm = transforms[args.task]
    t_dataset = datasets[args.task](phase=PHASE.train, transform=t_taransfm)
    v_dataset = datasets[args.task](phase=PHASE.validation, transform=v_transfm)

    model = models[args.task]()
    optimizer = Adam(model.parameters(), lr=args.lr)
    error = nn.CrossEntropyLoss()
    cartography = Cartography()
    t_metrics = [Acc(), cartography]
    v_metrics = [Acc()]
    loggers = [ConsoleLogger(), FileLogger(args.logdir)]
    savers = [best_model.MINMetricValueModelSaver(model, savedir=args.logdir)]
    
    t_loader =  DataLoader(
        dataset=t_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fns[args.task]
    )
    
    v_loader =  DataLoader(
        dataset=v_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fns[args.task]
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
    cartography.value.to_pickle(osp.join(args.logdir, "cartography.pkl"))


if __name__ == "__main__":
    main()
