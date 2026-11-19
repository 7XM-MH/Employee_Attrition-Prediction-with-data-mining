"""Modèles non supervisés pour l'analyse de l'attrition des employés.

Ce module est complémentaire au pipeline supervisé de ``scripts/train_pipeline``.
Aucun modèle ici n'utilise ``Attrition`` pendant l'ajustement : la cible sert
uniquement, après coup, à décrire les clusters obtenus. ``EmployeeID`` est
toujours exclu.

Modèles proposés :
    KMeans                  — partitionnement en k groupes
    PCA                     — réduction de dimension / projection 2D
    AgglomerativeClustering — hiérarchie ascendante (linkage ``ward``)
    DBSCAN                  — détection de densité et de points atypiques

Les clés de ``UNSUPERVISED_MODELS`` sont stables et peuvent être utilisées
telles quelles dans les rapports et notebooks.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .preprocessing import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    ID_COL,
    NUMERICAL_FEATURES,
    TARGET,
    load_raw_data,
)

RANDOM_STATE = 42
DEFAULT_N_CLUSTERS = 4
DEFAULT_DBSCAN_EPS = 0.18
DEFAULT_DBSCAN_MIN_SAMPLES = 25
SILHOUETTE_SAMPLE_SIZE = 2000

UNSUPERVISED_MODELS = {
    "KMeans": "Partitionnement des employés en groupes de profils similaires.",
    "PCA": "Projection des 12 caractéristiques en 2 axes de variance maximale.",
    "AgglomerativeClustering": "Construction hiérarchique de groupes par linkage de Ward.",
    "DBSCAN": "Regroupement par densité et repérage des employés atypiques.",
}


def build_unsupervised_preprocessor() -> ColumnTransformer:
    """Préprocesseur sans cible : numériques standardisés, catégorielles encodées."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def encode_features(df: pd.DataFrame | None = None) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """Retourne (matrice encodée, noms des colonnes, dataframe brut).

    Le préprocesseur est ajusté sur toutes les lignes : il n'y a pas d'étiquette
    à préserver ici. ``EmployeeID`` et ``Attrition`` ne sont jamais dans ``X``.
    """
    raw = load_raw_data() if df is None else df.copy()
    X = raw[ALL_FEATURES]
    encoder = build_unsupervised_preprocessor()
    encoded = encoder.fit_transform(X)
    matrix = np.asarray(encoded, dtype=float)
    names = [str(name) for name in encoder.get_feature_names_out()]
    return matrix, names, raw


def _silhouette(matrix: np.ndarray, labels: np.ndarray) -> float | None:
    """Silhouette calculée sur un échantillon, en excluant le bruit DBSCAN (-1)."""
    valid = labels != -1
    X_valid, y_valid = matrix[valid], labels[valid]
    n_labels = len(np.unique(y_valid))
    if n_labels < 2 or n_labels >= len(y_valid):
        return None

    sample_size = min(SILHOUETTE_SAMPLE_SIZE, len(X_valid))
    rng = np.random.default_rng(RANDOM_STATE)
    indices = rng.choice(len(X_valid), size=sample_size, replace=False)
    score = silhouette_score(X_valid[indices], y_valid[indices])
    return round(float(score), 4)


def _cluster_summary(
    labels: np.ndarray, raw: pd.DataFrame, matrix: np.ndarray
) -> dict[str, Any]:
    """Taille des clusters, bruit éventuel et taux d'attrition a posteriori."""
    valid = labels != -1
    counts = pd.Series(labels[valid]).value_counts().sort_index()
    summary: dict[str, Any] = {
        "n_clusters": int(counts.size),
        "cluster_sizes": {str(int(k)): int(v) for k, v in counts.items()},
        "silhouette": _silhouette(matrix, labels),
    }
    if np.any(~valid):
        summary["n_noise"] = int((~valid).sum())

    if TARGET in raw.columns:
        target = raw[TARGET].astype(str).to_numpy()
        rates = (
            pd.DataFrame({"cluster": labels[valid], TARGET: target[valid]})
            .groupby("cluster")[TARGET]
            .apply(lambda s: (s == "Yes").mean())
        )
        summary["attrition_rate_by_cluster"] = {
            str(int(k)): round(float(v), 4) for k, v in rates.items()
        }
    return summary


def run_pca(matrix: np.ndarray, feature_names: list[str], n_components: int = 2) -> dict[str, Any]:
    """Réduction de dimension et composantes principales interprétables."""
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    coordinates = pca.fit_transform(matrix)
    components: list[dict[str, Any]] = []
    for index in range(n_components):
        loadings = pca.components_[index]
        top = np.argsort(np.abs(loadings))[-3:][::-1]
        components.append({
            "component": index + 1,
            "explained_variance_ratio": round(float(pca.explained_variance_ratio_[index]), 4),
            "top_features": [
                {
                    "feature": feature_names[int(pos)],
                    "loading": round(float(loadings[int(pos)]), 4),
                }
                for pos in top
            ],
        })
    return {
        "model": pca,
        "coordinates": coordinates,
        "components": components,
        "total_explained_variance": round(float(pca.explained_variance_ratio_.sum()), 4),
    }


def run_kmeans(matrix: np.ndarray, raw: pd.DataFrame, n_clusters: int = DEFAULT_N_CLUSTERS) -> dict[str, Any]:
    """K-Means avec plusieurs initialisations pour stabiliser le résultat."""
    model = KMeans(n_clusters=n_clusters, n_init=10, random_state=RANDOM_STATE)
    labels = model.fit_predict(matrix)
    result = _cluster_summary(labels, raw, matrix)
    result.update({"model": model, "labels": labels, "inertia": round(float(model.inertia_), 2)})
    return result


def run_agglomerative(
    matrix: np.ndarray,
    raw: pd.DataFrame,
    n_clusters: int = DEFAULT_N_CLUSTERS,
    n_neighbors: int = 10,
) -> dict[str, Any]:
    """Clustering hiérarchique ascendant avec linkage de Ward.

    Une matrice de connectivité de k plus proches voisins évite de matérialiser
    la matrice de distance complète (10 000 x 10 000) sur ce jeu de données.
    """
    connectivity = kneighbors_graph(matrix, n_neighbors=n_neighbors, include_self=False)
    connectivity = connectivity.maximum(connectivity.T)
    model = AgglomerativeClustering(
        n_clusters=n_clusters, linkage="ward", connectivity=connectivity
    )
    labels = model.fit_predict(matrix)
    result = _cluster_summary(labels, raw, matrix)
    result.update({"model": model, "labels": labels})
    return result


def run_dbscan(
    coordinates: np.ndarray,
    raw: pd.DataFrame,
    eps: float = DEFAULT_DBSCAN_EPS,
    min_samples: int = DEFAULT_DBSCAN_MIN_SAMPLES,
) -> dict[str, Any]:
    """DBSCAN appliqué à la projection PCA 2D pour éviter la malédiction des dimensions."""
    model = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
    labels = model.fit_predict(coordinates)
    result = _cluster_summary(labels, raw, coordinates)
    result.update({"model": model, "labels": labels, "eps": eps, "min_samples": min_samples})
    return result


def analyse_unsupervised(
    df: pd.DataFrame | None = None,
    n_clusters: int = DEFAULT_N_CLUSTERS,
    eps: float = DEFAULT_DBSCAN_EPS,
    min_samples: int = DEFAULT_DBSCAN_MIN_SAMPLES,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Exécute les 4 modèles et retourne (résumé JSON, détails avec estimateurs).

    Le résumé ne contient que des types Python natifs : il peut être écrit
    directement dans ``results/unsupervised_summary.json``. Les détails donnent
    accès aux labels, aux coordonnées PCA et aux estimateurs pour les notebooks.
    """
    matrix, feature_names, raw = encode_features(df)

    pca = run_pca(matrix, feature_names)
    kmeans = run_kmeans(matrix, raw, n_clusters)
    agglomerative = run_agglomerative(matrix, raw, n_clusters)
    dbscan = run_dbscan(pca["coordinates"], raw, eps, min_samples)

    summary = {
        "dataset": {
            "rows": int(len(raw)),
            "input_features": len(ALL_FEATURES),
            "encoded_features": len(feature_names),
            "excluded_columns": [ID_COL, TARGET],
        },
        "preprocessing": {
            "numerical": "StandardScaler",
            "categorical": "OneHotEncoder(handle_unknown='ignore')",
            "target_used_for_fitting": False,
        },
        "models": list(UNSUPERVISED_MODELS),
        "PCA": {
            "n_components": 2,
            "total_explained_variance": pca["total_explained_variance"],
            "components": pca["components"],
        },
        "KMeans": {k: v for k, v in kmeans.items() if k not in ("model", "labels")},
        "AgglomerativeClustering": {
            k: v for k, v in agglomerative.items() if k not in ("model", "labels")
        },
        "DBSCAN": {k: v for k, v in dbscan.items() if k not in ("model", "labels")},
        "interpretation_note": (
            "Les taux d'attrition par cluster sont calculés après coup uniquement "
            "pour interpréter les groupes ; ils n'interviennent jamais dans l'ajustement."
        ),
    }
    details = {
        "matrix": matrix,
        "feature_names": feature_names,
        "raw": raw,
        "estimators": {
            "PCA": pca["model"],
            "KMeans": kmeans["model"],
            "AgglomerativeClustering": agglomerative["model"],
            "DBSCAN": dbscan["model"],
        },
        "labels": {
            "KMeans": kmeans["labels"],
            "AgglomerativeClustering": agglomerative["labels"],
            "DBSCAN": dbscan["labels"],
        },
        "pca_coordinates": pca["coordinates"],
    }
    return summary, details
