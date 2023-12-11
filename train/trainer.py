import time
import torch
import numpy as np
from copy import copy
from queue import Queue
from typing import List
from cProfile import label
from enums.phase import PHASE
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
from saver.saver_interface import IModelSaver
from logger.logger_interface import ILogger
from metric.metric_interface import IMetric
from sklearn.model_selection import KFold
from utils.loss_reduction import median


class Trainer:
    def __init__(
        self,
        model,
        error,
        device,
        t_loader,
        num_folds,
        optimizer,
        num_epochs,
        t_metrics: List[IMetric],
        v_metrics: List[IMetric],
        savers: List[IModelSaver],
        logQ: Queue,
        loss_reduction_method='mean'
    ) -> None:
        self.logQ = logQ
        self.model = model
        self.error = error
        self.device = device
        self.t_metrics = t_metrics
        self.v_metrics = v_metrics
        self.t_loader = t_loader
        self.num_folds = num_folds
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.savers = savers
        self.loss_reduction_method = loss_reduction_method

    def start(self):
        print('training start ...')
        self.model.to(self.device)

        kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=43)

        for fold, (train_indices, val_indices) in enumerate(kf.split(self.t_loader.dataset)):
            print(f"\nFold {fold + 1}/{self.num_folds}")
            t_dataset_fold = torch.utils.data.Subset(
                self.t_loader.dataset, train_indices)
            v_dataset_fold = torch.utils.data.Subset(
                self.t_loader.dataset, val_indices)

            t_loader_fold = DataLoader(
                dataset=t_dataset_fold,
                batch_size=self.t_loader.batch_size,
                shuffle=True,
                collate_fn=self.t_loader.collate_fn
            )

            v_loader_fold = DataLoader(
                dataset=v_dataset_fold,
                batch_size=self.t_loader.batch_size,
                shuffle=False,
                collate_fn=self.t_loader.collate_fn
            )

            for epoch in range(self.num_epochs):
                start_time = time.time()
                # train
                train_epoch_loss = []
                self.model.train()
                iteration = 0 
                for idx, data, labels in t_loader_fold:
                    self.optimizer.zero_grad()
                    data, labels = data.to(self.device), labels.to(self.device)
                    prediction_values = self.model(data)  # (B, C)
                    prediction_probs = softmax(prediction_values, dim=1)  # (B, C)
                    loss_each = self.error(prediction_probs, labels)
                    if self.loss_reduction_method == 'mean':
                        loss_all = torch.mean(loss_each)
                    elif self.loss_reduction_method == 'median':
                        loss_all = median(loss_each)
                    else:
                        raise Exception(f"Unknown loss reduction method {self.loss_reduction_method}")
                    train_epoch_loss.append(loss_all.item())
                    loss_all.backward()
                    self.optimizer.step()

                    train_result = (prediction_probs, labels, loss_each)
                    metric_results = []
                    for metric in self.t_metrics:
                        metric_results.append(metric.calculate(*train_result))

                    self.logQ.put({
                        "fold": fold, "epoch": epoch, "iteration": iteration,
                        "samples": copy(idx), "phase": PHASE.train,
                        "loss_reduction": self.loss_reduction_method,
                        "labels": np.argmax(labels.cpu().detach().numpy(), axis=1),
                        "metrics": metric_results
                    })
                    iteration += 1

                # validation
                validation_epoch_loss = []
                self.model.eval()
                iteration = 0 
                for idx, data, labels in v_loader_fold:
                    data, labels = data.to(self.device), labels.to(self.device)
                    prediction_values = self.model(data)  # (B, C)
                    prediction_probs = softmax(prediction_values, dim=1)  # (B, C)
                    loss_each = self.error(prediction_probs, labels)
                    if self.loss_reduction_method == 'mean':
                        loss_all = torch.mean(loss_each)
                    elif self.loss_reduction_method == 'median':
                        loss_all = median(loss_each)
                    else:
                        raise Exception(f"Unknown loss reduction method {self.loss_reduction_method}")
                    validation_epoch_loss.append(loss_all.item())

                    validation_result = (prediction_probs, labels, loss_each)
                    metric_results = []
                    for metric in self.v_metrics:
                        metric_results.append(metric.calculate(*validation_result))

                    self.logQ.put({
                        "fold": fold, "epoch": epoch, "iteration": iteration,
                        "samples": copy(idx), "phase": PHASE.validation,
                        "loss_reduction": self.loss_reduction_method,
                        "labels": np.argmax(labels.cpu().detach().numpy(), axis=1),
                        "metrics": metric_results
                    })
                    iteration += 1
                elapsed_time = time.time() - start_time

                print(f"epoch ({epoch}) duration ({time.strftime('%H:%M:%S', time.gmtime(elapsed_time))})| train-loss ({np.mean(train_epoch_loss)}) | val-loss ({np.mean(validation_epoch_loss)})")
                for saver in self.savers:
                    saver.look_for_save(metric_value=np.mean(validation_epoch_loss), epoch=epoch)

        self.logQ.put("EOF")
