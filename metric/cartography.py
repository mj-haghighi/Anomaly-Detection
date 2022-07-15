import torch
import pandas
import numpy as np
from typing import Dict, List
from train.dynamics import Dynamics
from enums import CARTOGRAPHY_METRICS
from .metric_interface import IMetric

SampleName = str
CartoMetrics = str # 'confidence', 'variability', 'correctness'
CartoData = Dict[str, List[float]]

class Cartography(IMetric):
    """
    Calculate 3 metrics:
    'confidence', 'variability', 'correctness'
    """

    def __init__(self) -> None:        
        self.sampels: Dict[SampleName, CartoData] = {}

    def calculate(self, dynamics: Dynamics, prediction_probs: torch.Tensor, labels: torch.Tensor, idx: List[str]):
        """
        Calculate metric
        inputs:
            dynamics: Training dynamics 'Dynamics'.
            prediction_probs: model output after softmax. (B:batch size, C: number of classes)
            labels: data labels in term of categorical. (B:batch size, C: number of classes)
        """
        
        preds, pred_indices = prediction_probs.max(dim=1)
        preds, pred_indices = preds.cpu().detach().numpy(), pred_indices.cpu().detach().numpy()
        labels = labels.argmax(dim=1).cpu().detach().numpy()

        filter = pred_indices == labels
        for i, id in enumerate(idx):
            if dynamics.epoch == 0:
                self.sampels[id] = {
                    "pred_prob": [],
                    "T": 0,
                }
            self.sampels[id]["T"] += 1 if filter[i] == True else 0
            self.sampels[id]["pred_prob"].append(preds[i])
    
    @property
    def value(self) -> pandas.DataFrame:
        name = self.sampels.keys()
        conf = [np.array(self.sampels[name]["pred_prob"]).mean() for name in self.sampels.keys()]
        std = [np.array(self.sampels[name]["pred_prob"]).std() for name in self.sampels.keys()]
        crness = [self.sampels[name]["T"] for name in self.sampels.keys()]

        data = {
            "NAME": name,
            CARTOGRAPHY_METRICS.CONFIDENCE: conf,
            CARTOGRAPHY_METRICS.VARIABILITY: std,
            CARTOGRAPHY_METRICS.CORRECTNESS: crness}

        return pandas.DataFrame(data=data)
    
    def __str__(self) -> str:
        return ""
