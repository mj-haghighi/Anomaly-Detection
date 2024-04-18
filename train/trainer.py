import os
import time
import torch
import copy
import numpy as np
import torch.nn as nn
import optimizer as predifined_optimizers
import model as predefined_models
from utils.verbose import verbose
from saver import best_model
from enums import PHASE, VERBOSE
from data.set import Subset
from data.set import GeneralDataset
from lrscheduler import get_lrscheduler
from metric.level1 import Loss, Proba
from data.transforms import transforms
from logger.dataframe import DataframeLogger
from torch.nn.functional import softmax
from sklearn.model_selection import KFold
from saver.last_model import LastEpochModelSaver

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
                "samples": copy.deepcopy(idx), "phase": PHASE.TRAIN,
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
                "samples": copy.deepcopy(idx), "phase": PHASE.VALIDATION,
                "labels": np.argmax(labels.cpu().detach().numpy(), axis=1),
                "metrics": metrics_result
            })
            verbose('VALIDATION Fold {} Epoch: {} Iter: {} Loss: {:.2f}'.format(fold, epoch, iteration, loss_all), VERBOSE.LEVEL_3)
            iteration += 1
    return np.mean(epoch_loss)



def train_fold(num_folds, num_epochs, fold, model_name, dataset_name, optimizer_name, learning_rate, num_classes, pretrain, label_column, logdir, lr_scheduler_name, queue):
    fold_start_time = time.time()
    model = predefined_models.get(name=model_name, num_classes=num_classes, pretrain=pretrain)
    model.to(device)
    optimizer = predifined_optimizers.get(name=optimizer_name, model=model, learning_rate=learning_rate)
    lr_scheduler = get_lrscheduler(optimizer, lr_scheduler_name)

    dataset = GeneralDataset(dataset_name=dataset_name, label_column=label_column, phase=PHASE.TRAIN, transform=None)
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=43)
    folding = list(kf.split(dataset))
    train_subset_indices, validation_subset_indices = folding[fold]
    
    train_transform, validation_transform = transforms[dataset_name]
    train_subset = Subset(dataset, train_subset_indices, transform=train_transform)
    validation_subset = Subset(dataset, validation_subset_indices, transform=validation_transform)

    trainloader = torch.utils.data.DataLoader(train_subset, batch_size=64, shuffle=True)
    validationloader = torch.utils.data.DataLoader(validation_subset, batch_size=64, shuffle=False)

    model_savers = [best_model.MINMetricValueModelSaver(savedir=logdir)]
    last_epoch_model_saver = LastEpochModelSaver(savedir=logdir)

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        train_loss = train_one_epoch(fold, epoch, model, trainloader, optimizer, queue)
        val_loss = validate_one_epoch(fold, epoch, model, validationloader, queue)
        lr_scheduler.take_step(metrics=val_loss)
        for saver in model_savers:
            saver.look_for_save(metric_value=val_loss, epoch=epoch, fold=fold, model=model)
        verbose('Fold {} Epoch: {}  TrainLoss: {:.2f} ValLoss: {:.2f} Time: {} s'.format(fold, epoch, train_loss, val_loss, time.time() - epoch_start_time), VERBOSE.LEVEL_2)

    last_epoch_model_saver.save(epoch=num_epochs-1, fold=fold, model=model)
    verbose('Model {} Fold {} Time: {} s'.format(model_name, fold, time.time() - fold_start_time), VERBOSE.LEVEL_1)
    queue.put(None)

def logger(logdir, model_name, optimizer_name, queue):
    dataframe_logger = DataframeLogger(logdir=logdir, metric_columns=[metric.name for metric in metrics], model_name=model_name, opt_name=optimizer_name)
    dataframe_logger.start(queue)
