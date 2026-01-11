import pandas as pd
from sklearn.decomposition import PCA
from datasci_automation.core.base import PipelineStep

class PCAStep(PipelineStep):
    def __init__(self, n_components=2):
        self.pca = PCA(n_components=n_components)

    def fit(self, X, y=None):
        self.pca.fit(X)
        return self

    def transform(self, X):
        data = self.pca.transform(X)
        cols = [f"PC{i+1}" for i in range(data.shape[1])]
        return pd.DataFrame(data, columns=cols, index=X.index)
