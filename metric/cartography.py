import torch
from .IMetric import IMetric
from train.dynamics import Dynamics


class Cartography(IMetric):
    """
    Calculate 3 metrics:
    'confidence', 'variability', 'correctness'
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate(self, dynamics: Dynamics, prediction_probs: torch.Tensor, labels: torch.Tensor, idx: str):
        """
        Calculate metric
        inputs:
            dynamics: Training dynamics 'Dynamics'.
            prediction_probs: model output after softmax. (B:batch size, C: number of classes)
            labels: data labels in term of categorical. (B:batch size, C: number of classes)
        """
        pass
