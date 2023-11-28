import torch
import numpy as np
from cProfile import label
from torch.nn.functional import softmax
from typing import List
from logger.logger_interface import ILogger
from metric.metric_interface import IMetric
from saver.saver_interface import IModelSaver
from enums.phase import PHASE


class Trainer:
    def __init__(
        self,
        model,
        error,
        device,
        t_loader,
        v_loader,
        optimizer,
        num_epochs,
        t_metrics: List[IMetric],
        v_metrics: List[IMetric],
        loggers: List[ILogger],
        savers: List[IModelSaver]
    ) -> None:

        self.model = model
        self.error = error
        self.device = device
        self.t_metrics = t_metrics
        self.v_metrics = v_metrics
        self.t_loader = t_loader
        self.v_loader = v_loader
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.loggers = loggers
        self.savers = savers

    def start(self):
        print('training start ...')
        self.model.to(self.device)
        for epoch in range(self.num_epochs):

            # train
            train_epoch_loss = []
            self.model.train()
            for idx, data, labels in self.t_loader:
                self.optimizer.zero_grad()
                data, labels = data.to(self.device), labels.to(self.device)
                prediction_values = self.model(data)  # (B, C)
                prediction_probs = softmax(prediction_values, dim=1)  # (B, C)
                loss_each = self.error(prediction_probs, labels)
                loss_all = torch.mean(loss_each)
                train_epoch_loss.append(loss_all.item())
                loss_all.backward()
                self.optimizer.step()

                train_result = (prediction_probs, labels, loss_each)
                for metric in self.t_metrics:
                    metric.calculate(*train_result)
                for logger in self.loggers:
                    logger.log(
                        epoch=epoch, samples=idx, phase=PHASE.train,
                        labels=np.argmax(labels.cpu().detach().numpy(), axis=1),
                        true_labels=[None for l in labels],
                        metrics=self.t_metrics)

            # validation
            validation_epoch_loss = []
            self.model.eval()
            for idx, data, labels in self.v_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                prediction_values = self.model(data)  # (B, C)
                prediction_probs = softmax(prediction_values, dim=1)  # (B, C)
                loss_each = self.error(prediction_probs, labels)
                loss_all = torch.mean(loss_each)
                validation_epoch_loss.append(loss_all.item())
                validation_result = (prediction_probs, labels, loss_each)
                for metric in self.v_metrics:
                    metric.calculate(*validation_result)
                for logger in self.loggers:
                    logger.log(
                        epoch=epoch, samples=idx, phase=PHASE.validation,
                        labels=np.argmax(labels.cpu().detach().numpy(), axis=0),
                        true_labels=[None for l in labels],
                        metrics=self.v_metrics)

            print(f"epoch ({epoch}) | train-loss ({np.mean(train_epoch_loss)}) | val-loss ({np.mean(validation_epoch_loss)})")
            for saver in self.savers:
                saver.look_for_save(metric_value=np.mean(validation_epoch_loss))
