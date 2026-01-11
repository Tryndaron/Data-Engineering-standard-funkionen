from abc import ABC, abstractmethod
import pandas as pd

class PipelineStep(ABC):
    """
    Base class for all pipeline steps.
    """

    @abstractmethod
    def fit(self, X: pd.DataFrame, y=None):
        pass

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        pass

    def fit_transform(self, X: pd.DataFrame, y=None):
        self.fit(X, y)
        return self.transform(X)
