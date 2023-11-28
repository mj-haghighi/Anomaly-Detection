import torch
from metric.metric_interface import IMetric


class Loss(IMetric):

    def __init__(self):
        pass

    def calculate(self, prediction_probs: torch.Tensor, labels: torch.Tensor, loss):
        """ 
        Calculate metric
        inputs:
            prediction_probs: model output after softmax. (B:batch size, C: number of classes)
            labels: data labels in term of categorical. (B:batch size, C: number of classes)
            loss: loss per sample
        """
        self.__loss = loss.cpu().detach().numpy()

    @property
    def value(self):
        return self.__loss

    @property
    def name(self):
        return "loss"
