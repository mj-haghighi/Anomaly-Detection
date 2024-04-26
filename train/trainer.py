import os
import time
import torch
import copy
import numpy as np
import pandas as pd
import torch.nn as nn
import optimizer as predifined_optimizers
import model as predefined_models
from utils.verbose import verbose
from saver import best_model
from enums import PHASE, VERBOSE
from configs import configs
from configs.general import EXPERIMENT_INFO_PATH, FILTERING_EXPERIMENT_INFO_PATH, FILTERING_EXPERIMENT_BASE_DIR, EXPERIMENT_BASE_DIR, EXPERIMENT_COLS
from data.set import Subset
from data.set import GeneralDataset
from lrscheduler import get_lrscheduler
from metric.level1 import Loss, Proba
from data.transforms import transforms
from logger.dataframe import DataframeLogger
from torch.nn.functional import softmax
from sklearn.model_selection import KFold
from saver.last_model import LastEpochModelSaver
from data.filtering_policy import get_data_filtering_policy 

metrics = [Loss(), Proba()]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
error = nn.CrossEntropyLoss(reduction='none')

def calculate_metrics(prediction_probs, labels, loss_each):
    metric_results = []
    for metric in metrics:
        metric_results.append(metric.calculate(prediction_probs, labels, loss_each))
    return metric_results


def train_one_epoch(fold, epoch, model, dataloader, optimizer, queue):
    model.train()
    epoch_loss = []
    iteration = 0 
    for idx, data, labels in dataloader:
        time.sleep(0.005)
        optimizer.zero_grad()

        data, labels = data.to(device), labels.to(device)
        prediction_values = model(data)  # (B, C)
        prediction_probs = softmax(prediction_values, dim=1)  # (B, C)

        loss_each = error(prediction_values, labels)
        loss_all = torch.mean(loss_each)
        epoch_loss.append(loss_all.item())
        loss_all.backward()
        optimizer.step()
        metrics_result = calculate_metrics(prediction_probs, labels, loss_each)
        queue.put({
                "fold": fold, "epoch": epoch, "iteration": iteration,
                "samples": copy.deepcopy(idx.detach().cpu().numpy()), "phase": PHASE.TRAIN,
                "labels": np.argmax(labels.cpu().detach().numpy(), axis=1),
                "metrics": metrics_result
                })
        verbose('TRAIN Fold {} Epoch: {} Iter: {} Loss: {:2f}'.format(fold, epoch, iteration, loss_all), VERBOSE.LEVEL_3)
        iteration += 1
    return np.mean(epoch_loss)

def validate_one_epoch(fold, epoch, model, dataloader, queue):
    model.eval()
    epoch_loss = []
    iteration = 0
    with torch.no_grad():
        for idx, data, labels in dataloader:
            time.sleep(0.015)
            data, labels = data.to(device), labels.to(device)
            prediction_values = model(data)  # (B, C)
            prediction_probs = softmax(prediction_values, dim=1)  # (B, C)

            loss_each = error(prediction_values, labels)
            loss_all = torch.mean(loss_each)
            epoch_loss.append(loss_all.item())

            metrics_result = calculate_metrics(prediction_probs, labels, loss_each)
            if queue.qsize() > 30:
                while queue.qsize() > 1:
                    time.sleep(0.01)
            queue.put({
                "fold": fold, "epoch": epoch, "iteration": iteration,
                "samples": copy.deepcopy(idx.detach().cpu().numpy()), "phase": PHASE.VALIDATION,
                "labels": np.argmax(labels.cpu().detach().numpy(), axis=1),
                "metrics": metrics_result
            })
            verbose('VALIDATION Fold {} Epoch: {} Iter: {} Loss: {:.2f}'.format(fold, epoch, iteration, loss_all), VERBOSE.LEVEL_3)
            iteration += 1
    return np.mean(epoch_loss)



def train_fold(fold, queue, experiment_number, filtering_policy=None, based_on=None, experiment_dir=None):
    data_filtering_policy = None
    base_dir = EXPERIMENT_BASE_DIR
    if filtering_policy is not None:
        filtering_experiments = pd.read_csv(FILTERING_EXPERIMENT_INFO_PATH, index_col='index')
        data_filtering_policy = get_data_filtering_policy(
            policy_name=filtering_policy,
            metric_name=based_on,
            experiment_base_dir=EXPERIMENT_BASE_DIR,
            experiment_number=experiment_number, 
            experiments_info_path=EXPERIMENT_INFO_PATH,
            experiments_dataset_columns=EXPERIMENT_COLS)

    experiments       = pd.read_csv(EXPERIMENT_INFO_PATH, index_col='index')
    target_experiment = experiments.iloc[experiment_number]
    num_folds         = target_experiment['folds']
    num_epochs        = target_experiment['epochs']
    model_name        = target_experiment['model']
    dataset_name      = target_experiment['dataset']
    optimizer_name    = target_experiment['optim']
    learning_rate     = float(target_experiment['lr'][3:])
    pretrain          = target_experiment['init'] == 'pretrain'
    num_classes       = len(configs[dataset_name].classes)
    label_column      = f"noisy_label[{target_experiment['np']},{target_experiment['ns']}]" if float(target_experiment['np'][3:]) > 0.001 else 'true_label'
    lr_scheduler_name = target_experiment['lr_scheduler']

    fold_start_time = time.time()
    model = predefined_models.get(name=model_name, num_classes=num_classes, pretrain=pretrain)
    model.to(device)
    optimizer = predifined_optimizers.get(name=optimizer_name, model=model, learning_rate=learning_rate)
    lr_scheduler = get_lrscheduler(optimizer, lr_scheduler_name)

    dataset = GeneralDataset(
        dataset_name=dataset_name, label_column=label_column, transform=None,
        phase=PHASE.TRAIN, data_filtering_policy=data_filtering_policy)

    kf = KFold(n_splits=num_folds, shuffle=True, random_state=43)
    folding = list(kf.split(dataset))
    train_subset_indices, validation_subset_indices = folding[fold]
    
    train_transform, validation_transform = transforms[dataset_name]
    train_subset = Subset(dataset, train_subset_indices, transform=train_transform)
    validation_subset = Subset(dataset, validation_subset_indices, transform=validation_transform)

    trainloader = torch.utils.data.DataLoader(train_subset, batch_size=64, shuffle=True)
    validationloader = torch.utils.data.DataLoader(validation_subset, batch_size=64, shuffle=False)

    model_savers = [best_model.MINMetricValueModelSaver(savedir=experiment_dir)]
    last_epoch_model_saver = LastEpochModelSaver(savedir=experiment_dir)

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        train_loss = train_one_epoch(fold, epoch, model, trainloader, optimizer, queue)
        val_loss = validate_one_epoch(fold, epoch, model, validationloader, queue)
        if lr_scheduler:
            lr_scheduler.take_step(metrics=val_loss)
        for saver in model_savers:
            saver.look_for_save(metric_value=val_loss, epoch=epoch, fold=fold, model=model)
        verbose('Fold {} Epoch: {}  TrainLoss: {:.2f} ValLoss: {:.2f} Time: {:.2f} s'.format(fold, epoch, train_loss, val_loss, time.time() - epoch_start_time), VERBOSE.LEVEL_2)

    last_epoch_model_saver.save(epoch=num_epochs-1, fold=fold, model=model)
    verbose('Model {} Fold {} Time: {:.2f} s'.format(model_name, fold, time.time() - fold_start_time), VERBOSE.LEVEL_1)
    queue.put(None)

def logger(logdir, queue):
    dataframe_logger = DataframeLogger(logdir=logdir, metric_columns=[metric.name for metric in metrics])
    dataframe_logger.start(queue)
