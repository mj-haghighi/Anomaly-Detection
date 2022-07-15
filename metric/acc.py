import torch
import numpy as np
from typing import List
from train.dynamics import Dynamics

class Acc:
    def __init__(self) -> None:
        pass
        self.T = 0
        self.F = 0
    
    def calculate(self, dynamics:Dynamics, prediction_probs:torch.Tensor, labels: torch.Tensor, idx: List[str]):
        """ 
        Calculate metric
        inputs:
            dynamics: Training dynamics 'Dynamics'.
            prediction_probs: model output after softmax. (B:batch size, C: number of classes)
            labels: data labels in term of categorical. (B:batch size, C: number of classes)
        """
        preds = prediction_probs.argmax(dim=1).cpu().detach().numpy()
        labels = labels.argmax(dim=1).cpu().detach().numpy()

        if dynamics.iteration == 0:
            self.T, self.F = 0, 0

        filter = preds==labels
        self.T += np.sum(filter)
        self.F += np.sum(~filter)

    @property
    def value(self):
        return (self.T / (self.T + self.F))


    def __str__(self) -> str:
        return "acc: {:.4}".format(self.value)