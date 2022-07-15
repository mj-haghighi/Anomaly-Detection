import torch
from typing import List
from train.dynamics import Dynamics

class IMetric:
    def __init__(self) -> None:
        self.value = 0

    def calculate(self, dynamics:Dynamics, prediction_probs:torch.Tensor, labels: torch.Tensor, idx: List[str]):
        """ 
        Calculate metric
        inputs:
            dynamics: Training dynamics 'Dynamics'.
            prediction_probs: model output after softmax. (B:batch size, C: number of classes)
            labels: data labels in term of categorical. (B:batch size, C: number of classes)
        """
        raise Exception("This methoud is not implemented")
