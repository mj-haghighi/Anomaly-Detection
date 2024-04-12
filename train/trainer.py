import time
import torch
import numpy as np
from copy import copy
from queue import Queue
from typing import List
from cProfile import label
from data.set import Subset
from enums.phase import PHASE
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
from saver.saver_interface import IModelSaver
from logger.logger_interface import ILogger
from metric.metric_interface import IMetric
from sklearn.model_selection import KFold


class Trainer:
    def __init__(
        self,
        model,
        error,
        device,
        train_dataloader,
        validation_dataloader,
        num_folds,
        optimizer,
        lr_scheduler,
        num_epochs,
        t_metrics,
        v_metrics,
        savers,
        logQ,
        fold
    ) -> None:
        self.model = model
        self.error = error
        self.device = device
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.num_folds = num_folds
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.num_epochs = num_epochs
        self.t_metrics = t_metrics
        self.v_metrics = v_metrics
        self.savers = savers
        self.logQ = logQ
        self.fold = fold


    def start(self):
        print('training start ...')
        self.model.to(self.device)
        for epoch in range(self.num_epochs):
            start_time = time.time()
            # train
            train_epoch_loss = []
            self.model.train()
            iteration = 0 
            for idx, data, labels in self.train_dataloader:
                self.optimizer.zero_grad()
                data, labels = data.to(self.device), labels.to(self.device)
                prediction_values = self.model(data)  # (B, C)
                loss_each = self.error(prediction_values, labels)
                prediction_probs = softmax(prediction_values, dim=1)  # (B, C)
                loss_all = torch.mean(loss_each)
                train_epoch_loss.append(loss_all.item())
                loss_all.backward()
                self.optimizer.step()

                train_result = (prediction_probs, labels, loss_each)
                metric_results = []
                for metric in self.t_metrics:
                    metric_results.append(metric.calculate(*train_result))

                self.logQ.put({
                    "fold": self.fold, "epoch": epoch, "iteration": iteration,
                    "samples": copy(idx), "phase": PHASE.train,
                    "labels": np.argmax(labels.cpu().detach().numpy(), axis=1),
                    "metrics": metric_results
                })
                iteration += 1

            # validation
            with torch.no_grad():
                validation_epoch_loss = []
                self.model.eval()
                iteration = 0 
                for idx, data, labels in self.validation_dataloader:
                    data, labels = data.to(self.device), labels.to(self.device)
                    prediction_values = self.model(data)  # (B, C)
                    loss_each = self.error(prediction_values, labels)
                    prediction_probs = softmax(prediction_values, dim=1)  # (B, C)
                    loss_all = torch.mean(loss_each)
                    validation_epoch_loss.append(loss_all.item())

                    validation_result = (prediction_probs, labels, loss_each)
                    metric_results = []
                    for metric in self.v_metrics:
                        metric_results.append(metric.calculate(*validation_result))

                    self.logQ.put({
                        "fold": self.fold, "epoch": epoch, "iteration": iteration,
                        "samples": copy(idx), "phase": PHASE.validation,
                        "labels": np.argmax(labels.cpu().detach().numpy(), axis=1),
                        "metrics": metric_results
                    })
                    iteration += 1
                elapsed_time = time.time() - start_time

                print(f"epoch ({epoch}) duration ({time.strftime('%H:%M:%S', time.gmtime(elapsed_time))})| train-loss ({np.mean(train_epoch_loss)}) | val-loss ({np.mean(validation_epoch_loss)})")
                for saver in self.savers:
                    saver.look_for_save(metric_value=np.mean(validation_epoch_loss), epoch=epoch, fold=self.fold, model=self.model)

            if self.lr_scheduler is not None:
                self.lr_scheduler.take_step(metrics=np.mean(validation_epoch_loss))

        self.logQ.put("EOF")
