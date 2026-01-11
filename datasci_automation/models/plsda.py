import pandas as pd
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from datasci_automation.core.base import PipelineStep

class PLSDAClassifier(PipelineStep):
     """
    Partial Least Squares Discriminant Analysis (PLS-DA) Klassifikator.

    PLS-DA ist ein überwachtes Verfahren zur Dimensionsreduktion und Klassifikation.
    Es projiziert hochdimensionale Eingabedaten (X) in einen niedrigdimensionalen
    latenten Raum, der die Kovarianz zwischen den Merkmalen (X) und den Klassenlabels
    (y) maximiert. Die resultierenden latenten Variablen sind so optimiert, dass sie
    die Klassen möglichst gut voneinander trennen.

    Diese Implementierung verwendet intern ein PLS-Regressionsmodell, dessen
    kontinuierliche Ausgaben als Klassenscores interpretiert werden.

    Typische Anwendungsfälle:
    - Omics-Daten (Metabolomik, Proteomik, Spektroskopie)
    - Hochdimensionale Datensätze (viele Features, wenige Proben)
    - Stark korrelierte Eingangsvariablen

    Voraussetzungen:
    - X muss ein numerisches pandas DataFrame sein
    - y muss Klassenlabels enthalten (z. B. Integer oder One-Hot-Kodierung)

    Attribute:
        model (PLSRegression): Intern verwendetes PLS-Regressionsmodell

    Hinweise:
    - PLS-DA neigt stark zu Overfitting, wenn die Anzahl der Features deutlich größer
      ist als die Anzahl der Samples. Eine Kreuzvalidierung ist zwingend empfohlen.
    - Die Vorhersage erfolgt über die Klasse mit dem höchsten vorhergesagten Score.

    Beispiel:
        >>> clf = PLSDAClassifier(n_components=3)
        >>> clf.fit(X_train, y_train)
        >>> preds = clf.predict(X_test)
    """
     
    def __init__(self, n_components=2):
        self.model = PLSRegression(n_components=n_components)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def transform(self, X):
        return X  # classifier does not transform

    def predict(self, X):
        preds = self.model.predict(X)
        return np.argmax(preds, axis=1)
