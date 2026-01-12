import pandas as pd
import numpy as np
from typing import Dict
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from datasci_automation.core.base import PipelineStep


class KMeansSilhouetteStep(PipelineStep):
    """
    Pipeline Step for KMeans clustering with automatic k selection using Silhouette Score.
    Adds a 'cluster' column to the DataFrame.
    """

    def __init__(
        self,
        k_min: int = 2,
        k_max: int = 10,
        scale: bool = True,
        random_state: int = 42,
        cluster_column: str = "cluster"
    ):
        self.k_min = k_min
        self.k_max = k_max
        self.scale = scale
        self.random_state = random_state
        self.cluster_column = cluster_column

        self.scaler = None
        self.model = None
        self.best_k = None
        self.silhouette_scores: Dict[int, float] = {}

    # --------------------------------------------------

    def fit(self, X: pd.DataFrame, y=None):
        X_num = X.select_dtypes(include=[np.number])

        if X_num.empty:
            raise ValueError("KMeans requires numeric columns.")

        if self.scale:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_num)
        else:
            X_scaled = X_num.values

        best_score = -1

        for k in range(self.k_min, self.k_max + 1):
            model = KMeans(
                n_clusters=k,
                random_state=self.random_state,
                n_init=10
            )
            labels = model.fit_predict(X_scaled)

            score = silhouette_score(X_scaled, labels)
            self.silhouette_scores[k] = score

            if score > best_score:
                best_score = score
                self.best_k = k
                self.model = model

        return self

    # --------------------------------------------------

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        X_num = X.select_dtypes(include=[np.number])

        if self.scale:
            X_scaled = self.scaler.transform(X_num)
        else:
            X_scaled = X_num.values

        clusters = self.model.predict(X_scaled)
        X_out[self.cluster_column] = clusters

        return X_out

    # --------------------------------------------------

    def predict(self, X: pd.DataFrame):
        X_num = X.select_dtypes(include=[np.number])

        if self.scale:
            X_scaled = self.scaler.transform(X_num)
        else:
            X_scaled = X_num.values

        return self.model.predict(X_scaled)

    # --------------------------------------------------

    def get_summary(self) -> dict:
        return {
            "best_k": self.best_k,
            "silhouette_scores": self.silhouette_scores,
            "final_silhouette": self.silhouette_scores.get(self.best_k)
        }
