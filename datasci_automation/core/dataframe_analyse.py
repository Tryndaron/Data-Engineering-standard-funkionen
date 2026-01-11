import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict
from sklearn.cluster import KMeans
from typing import Iterable, Dict, Any
from sklearn.cross_decomposition import PLSRegression



def pca_tranfo(X: pd.DataFrame,
                        variance_threshold: float = 0.95,
                        scale: bool = True):
    """This function performs a PCA transformation with the right amount of 
    PC for a variance of above 95%.

    Args:
        X (pd.DataFrame): _description
        variance_threshold (float, optional): _description_. Defaults to 0.95.
        scale (bool, optional): _description_. Defaults to True.

    Raises:
        TypeError: _description_
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    
    # Validierung
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X muss ein pandas DataFrame sein")
    
    if X.select_dtypes(include="number").shape[1] != X.shape[1]:
        raise ValueError("X darf nur numerische Spalten enthalten")  
    
    #Skalierung
    scaler = StandardScaler() if scale else None
    X_scaled = scaler.fit_transform(X) if scale else X.values

    #PCA mit allen Komponenten
    pca_full = PCA()
    pca_full.fit(X_scaled)

    #optimale Komponenten bestimmen
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    optimal_components = np.argmax(cumulative_variance >= variance_threshold) + 1

    # PCA mit optimaler Komponentenanzahl
    pca = PCA(n_components=optimal_components)
    X_pca = pca.fit_transform(X_scaled)

    #Ergebnisse als DataFrame

    columns = [f"PC{i+1} " for i in range(optimal_components)]
    X_pca_df = pd.DataFrame(X_pca, columns=columns, index=X.index)

    #Metadata
    info = {
        "optimal_components": optimal_components,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_variance": np.cumsum(pca.explained_variance_ratio_),
        #"variance_threshold": variance_threshold,
        "scaler_used": scale
    }  
    return X_pca_df, pca, info




########################################################################################################################




def fit_kmeans_dataframe(
        df: pd.DataFrame,
        n_clusters: int,
        cluster_column: str= "cluster",
        scale: bool = True,
        random_state: int= 42,
        visualize: bool=False,
) -> Tuple[pd.DataFrame, KMeans]:
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        n_clusters (int): _description_
        cluster_column (str, optional): _description_. Defaults to "cluster".
        scale (bool, optional): _description_. Defaults to True.
        random_state (int, optional): _description_. Defaults to 42.

    Returns:
        Tuple[pd.DataFrame, KMeans]: _description_
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df muss ein pandas Dataframe sein")
    
    X = df.select_dtypes(include="number").copy()
    if X.empty:
        raise ValueError("DataFrame muss numerische Spalten enthalten")

    #Skalierung
    if scale:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X.values



    #KMeans fitten
    kmeans = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=10,
        random_state=random_state,
    )
    
    labels = kmeans.fit_predict(X_scaled)

    #Ergebnis zurück in DataFrame schreiben

    df_clustered = df.copy()
    df_clustered[cluster_column] = labels

    pca = None

    #Visualisieren mit pca = 2

    if visualize:
        pca = PCA(n_components=2, random_state=random_state)
        X_pca = pca.fit_transform(X_scaled)
        z =[] 
        for i in df.index:
            z.append(str(i))

        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111)
        plt.scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            c=labels,  
        )
        #plt.annotate()

        plt.xlabel("PCA Komponente 1")
        plt.ylabel("PCA Komponente 2")
        plt.title("KMeans Cluster (PCA 2D)")
        plt.colorbar(label="Cluster")
        #plt.text(fontsize=12)
        plt.grid(True)

        for i, txt in enumerate(z):
            ax.annotate(txt, (X_pca[:, 0][i],  X_pca[:, 1][i] ), fontsize=12)



        plt.show()

    return df_clustered, kmeans, pca



def korrelation_analysis(df: pd.DataFrame, min_abs_corr: float= 0.0):
    """Dies Funkion führt eine komplette Korrelationsanalysis eines pandas 
    Dataframes dar.

    Args:
        df (pd.DataFrame): Pandas Dataframe was analysiert werden soll
    """
    #es dürfen nur numerische Werte verwendet werden, daher werden nicht 
    #numerische Werte hier aussortiert
    numeric_df = df.select_dtypes(include="number")

    #Berechnen der Korrelation jeder SPalte
    corr_matrix = numeric_df.corr(method="pearson")

    #Long Format 
    corr_long = (
        corr_matrix.stack().reset_index().rename(columns= {
            "level_0": "feature_1",
            "level_1": "feature_2",
            0: "correlation"
        } )
    )

    # Selbstkorrelationen entfernen
    corr_long = corr_long[corr_long["feature_1"] != corr_long["feature_2"]]

    #Doppelte Paare entfernen
    corr_long["sorted_pair"] = corr_long.apply(
        lambda x: tuple(sorted([x["feature_1"], x["feature_2"]])),
        axis=1
    )
    corr_long = corr_long.drop_duplicates(subset="sorted_pair")
    corr_long = corr_long.drop(columns="sorted_pair")

    #Nach Mindestkorrelation filtern
    corr_long = corr_long[
        corr_long["correlation"].abs() >= min_abs_corr 
    ].sort_values(
        by="correlation", key=abs, ascending=False
    )

    return corr_long.reset_index(drop=True) 


#################################################################################

def plot_correlation_matrix(df: pd.DataFrame):
    ""
    numeric_df = df.select_dtypes(include="number")
    corr = numeric_df.corr(method="pearson")

    plt.figure(figsize=(40, 32))
    plt.imshow(corr)
    plt.colorbar()
    plt.xticks(range(len(corr)), corr.columns, rotation=90)
    plt.yticks(range(len(corr)), corr.columns)
    plt.title("Korrelationsmatrix")
    plt.tight_layout()
    plt.show()


################################################################################


def biplot(
        df: pd.DataFrame,
        n_components: int=2,
        scale: bool=True,
        figsize: tuple = (16,12)
):
    #nuermische Spalten werden akzeptiert
    X = df.select_dtypes(include="number")

    if X.shape[1] < 2:
        raise ValueError("Mindestens zwei numerische Spalten müssen im Dataframe vorliegen !")

    #Standardisieren
    if scale:
        x_scaled = StandardScaler().fit_transform(X)
    else:
        x_scaled = X.values
    # PCA wird mit zwei komponente durchgeführt
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(x_scaled)
    loadings = pca.components_.T

    #Plot 
    plt.figure(figsize=figsize)

    #Datenpunkte
    plt.scatter(scores[:,0], scores[:, 1], alpha=0.7)

    #Loadings (feature Vektoren)
    for i, feature in enumerate(X.columns):
        plt.arrow(
            0, 0,
            loadings[i, 0],
            loadings[i, 1],
            color="red",
            alpha=0.6,
            head_width=0.03  
        )
        """ plt.text(
            loadings[i, 0] * 1.15,
            loadings[i, 1] * 1.15,
            feature,
            color="red",
            ha="center",
            va="center"  
        ) """
    #Achsen & Labels
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
   # plt.axhline(0, color="grey", linewidth=0.5)
    #plt.axvline(0, color="grey", linewidth=0.5)
    plt.title("PCA Biplot")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


#################################################################################




from sklearn.cross_decomposition import PLSRegression


def pls_da(
    df: pd.DataFrame,
    labels: Iterable,
    n_components: int = 2
) -> Dict[str, Any]:
    """
    Führt PLS-DA (Partial Least Squares Discriminant Analysis) auf einem
    pandas DataFrame durch. Als Rückgabewert erhält man ein Dictionary mit
    Modell, Scores, Loadings und VIP-Scores. Modell ist der PLSRegression-Objekt, Scores sind die PLS-Scores
    und Loadings sind die PLS-Ladungen. VIP-Scores sind die Variablenwichtigkeitsscores.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-Daten (nur numerische Spalten)
    labels : Iterable
        Klassenlabels (z.B. Liste, pd.Series, np.ndarray)
    n_components : int
        Anzahl der PLS-Komponenten

    Returns
    -------
    Dict[str, Any]
        Dictionary mit Modell, Scores, Loadings und VIP-Scores
    """

    # ---------- Features ----------
    X: np.ndarray = df.values
    feature_names: pd.Index = df.columns

    # ---------- Labels ----------
    encoder: LabelEncoder = LabelEncoder()
    y: np.ndarray = encoder.fit_transform(labels).reshape(-1, 1)

    # ---------- Skalierung ----------
    scaler: StandardScaler = StandardScaler()
    X_scaled: np.ndarray = scaler.fit_transform(X)

    # ---------- PLS-DA ----------
    pls: PLSRegression = PLSRegression(n_components=n_components)
    X_scores: np.ndarray
    Y_scores: np.ndarray
    X_scores, Y_scores = pls.fit_transform(X_scaled, y)

    # ---------- Loadings ----------
    loadings: pd.DataFrame = pd.DataFrame(
        pls.x_loadings_,
        index=feature_names,
        columns=[f"PLS{i+1}" for i in range(n_components)]
    )

    # ---------- VIP Scores ----------
    T: np.ndarray = pls.x_scores_
    W: np.ndarray = pls.x_weights_
    Q: np.ndarray = pls.y_loadings_

    p: int
    h: int
    p, h = W.shape

    s: np.ndarray = np.sum((T @ Q.T) ** 2, axis=0)

    vip: np.ndarray = np.sqrt(
        p * (W ** 2 @ s) / np.sum(s)
    )

    vip_scores: pd.Series = pd.Series(
        vip.flatten(),
        index=feature_names,
        name="VIP"
    ).sort_values(ascending=False)

    return {
        "model": pls,
        "scores": pd.DataFrame(
            X_scores,
            columns=[f"PLS{i+1}" for i in range(n_components)]
        ),
        "loadings": loadings,
        "vip_scores": vip_scores,
        "scaler": scaler,
        "label_encoder": encoder
    }




