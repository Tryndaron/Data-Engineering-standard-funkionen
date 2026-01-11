import pandas as pd
from sklearn.preprocessing import StandardScaler
from datasci_automation.core.base import PipelineStep

class StandardizeStep(PipelineStep):
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        self.columns = X.columns
        self.scaler.fit(X)
        return self

    def transform(self, X):
        data = self.scaler.transform(X)
        return pd.DataFrame(data, columns=self.columns, index=X.index)
