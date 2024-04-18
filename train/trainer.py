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
        print('TRAIN Fold %d Epoch: %d Iter: %d' % (fold, epoch, iteration))
        iteration += 1

def validate_one_epoch(fold, epoch, model, dataloader, optimizer, queue):
    model.eval()
    epoch_loss = []
    iteration = 0 
    for idx, data, labels in dataloader:
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
            "samples": copy.deepcopy(idx), "phase": PHASE.VALIDATION,
            "labels": np.argmax(labels.cpu().detach().numpy(), axis=1),
            "metrics": metrics_result
        })
        print('VALIDATION Fold %d Epoch: %d Iter: %d' % (fold, epoch, iteration))
        iteration += 1
    return np.mean(epoch_loss)



def train_fold(num_folds, num_epochs, fold, model_name, dataset_name, optimizer_name, learning_rate, num_classes, pretrain, label_column, logdir, lr_scheduler_name, queue):
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

    for epoch in range(num_epochs):
        # print("Im here2")
        train_one_epoch(fold, epoch, model, trainloader, optimizer, queue)
        # print("Im here3")
        loss = validate_one_epoch(fold, epoch, model, validationloader, optimizer, queue)
        # print("Im here4") 
        lr_scheduler.take_step(metrics=loss)
        # print("Im here5")
        for saver in model_savers:
        #     print("Im here6")
            saver.look_for_save(metric_value=loss, epoch=epoch, fold=fold, model=model)
    print(f"Fold {fold} Finish!")
    queue.put(None)

def logger(logdir, model_name, optimizer_name, queue):
    dataframe_logger = DataframeLogger(logdir=logdir, metric_columns=[metric.name for metric in metrics], model_name=model_name, opt_name=optimizer_name)
    dataframe_logger.start(queue)
