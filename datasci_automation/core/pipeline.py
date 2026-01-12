import pandas as pd
from typing import List
from datasci_automation.core.base import PipelineStep
from datasci_automation.reporting.summary import dataframe_summary

class DataPipeline:
    def __init__(self, steps: List[PipelineStep]):
        self.steps = steps

    def fit(self, X: pd.DataFrame, y=None):
        self.summary = dataframe_summary(X)
        for step in self.steps:
            X = step.fit_transform(X, y)
        self._last_X = X
        return self

    def transform(self, X: pd.DataFrame):
        for step in self.steps:
            X = step.transform(X)
        return X

    def predict(self, X: pd.DataFrame):
        for step in self.steps[:-1]:
            X = step.transform(X)
        return self.steps[-1].predict(X)

    def run(self, X: pd.DataFrame, y=None):
        self.fit(X, y)
        return self._last_X
