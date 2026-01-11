import numpy as np
import pandas as pd
from datasci_automation.core.base import PipelineStep

class NanImputerStep(PipelineStep):
    """
    Ersetzt fehlende Werte (NaN) durch den Mittelwert jeder Spalte.
    """

    def fit(self, X, y=None):
        self.means = X.mean()
        return self

    def transform(self, X):
        return X.fillna(self.means)
    
