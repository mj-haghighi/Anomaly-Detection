from torch.nn.functional import softmax
from typing import List
from logger.logger_interface import ILogger
from metric.metric_interface import IMetric
from .dynamics import Dynamics

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
        loggers: List[ILogger]
        ) -> None:

        self.model = model
        self.error = error
        self.device = device
        self.t_metrics = t_metrics
        self.v_metrics = v_metrics
        self.t_loader = t_loader
        self.v_loader = v_loader
        self.t_dynamics = Dynamics()
        self.v_dynamics = Dynamics()
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.loggers = loggers

    def start(self):
        print('training start ...')
        self.model.to(self.device)
        for epoch in range(self.num_epochs):
            self.t_dynamics.epoch = epoch
            self.t_dynamics.iteration = -1
            self.v_dynamics.epoch = epoch
            self.v_dynamics.iteration = -1

            # train            
            self.model.train()
            for idx, data, labels in self.t_loader:
                self.t_dynamics.iteration +=1
                self.optimizer.zero_grad()
                data = data.to(self.device)
                prediction_values = self.model(data) # (B, C)
                prediction_probs = softmax(prediction_values, dim=1) # (B, C)
                loss = self.error(prediction_probs, labels)
                self.t_dynamics.b_loss = loss.item()
                loss.backward()
                self.optimizer.step()
            
                train_result = (prediction_probs, labels, idx)
                for metric in self.t_metrics:
                    metric.calculate(self.t_dynamics, *train_result)
                
            # validation
            self.model.eval()
            for idx, data, labels in self.v_loader:
                self.v_dynamics.iteration +=1
                data = data.to(self.device)
                prediction_values = self.model(data) # (B, C)
                prediction_probs = softmax(prediction_values, dim=1) # (B, C)
                loss = self.error(prediction_probs, labels)
                self.v_dynamics.b_loss = loss.item()

                validation_result = (prediction_probs, labels, idx)
                for metric in self.v_metrics:
                    metric.calculate(self.v_dynamics, *validation_result)

            for logger in self.loggers:
                logger.log(self.t_dynamics, self.v_dynamics, self.t_metrics, self.v_metrics)