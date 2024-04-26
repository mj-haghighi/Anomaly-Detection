from abc import ABC, abstractmethod
import pandas as pd

class FilteringPolicy(ABC):

    @abstractmethod
    def filter_samples(self) -> pd.DataFrame:
        pass