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
    KMedoids                — PAM simplifié (médoïdes = points réels du jeu)
    DIANA                   — clustering divisif par scissions successives
    CosineKMeans            — K-Means en géométrie cosinus (matrice L2-normalisée)

Les clés de ``UNSUPERVISED_MODELS`` sont stables et peuvent être utilisées
telles quelles dans les rapports et notebooks.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import OneHotEncoder, StandardScaler, normalize

from .preprocessing import (
    CATEGORICAL_FEATURES,
    FEATURE_COLS,
    ID_COL,
    NUMERICAL_FEATURES,
    TARGET,
    load_raw,
)

RANDOM_STATE = 42
DEFAULT_N_CLUSTERS = 4
DEFAULT_DBSCAN_EPS = 0.18
DEFAULT_DBSCAN_MIN_SAMPLES = 25
SILHOUETTE_SAMPLE_SIZE = 2000
DEFAULT_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "Employee_Attrition_DataSet.csv"
MEDOID_CANDIDATE_SAMPLE = 300
DENDROGRAM_SAMPLE_SIZE = 2000

UNSUPERVISED_MODELS = {
    "KMeans": "Partitionnement des employés en groupes de profils similaires.",
    "PCA": "Projection des 12 caractéristiques en 2 axes de variance maximale.",
    "AgglomerativeClustering": "Construction hiérarchique de groupes par linkage de Ward.",
    "DBSCAN": "Regroupement par densité et repérage des employés atypiques.",
    "KMedoids": "PAM simplifié : médoïdes pris parmi les employés réels du jeu.",
    "DIANA": "Clustering divisif : scissions successives du cluster au plus grand diamètre.",
    "CosineKMeans": "K-Means sur vecteurs L2-normalisés (géométrie de la similarité cosinus).",
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
    raw = load_raw(DEFAULT_DATA_PATH) if df is None else df.copy()
    X = raw[FEATURE_COLS]
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
            "input_features": len(FEATURE_COLS),
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


# --------------------------------------------------------------------------- #
# LOT 3 — clustering avancé (chapitres 7 à 10 du cours)
# --------------------------------------------------------------------------- #
def davies_bouldin(matrix: np.ndarray, labels: np.ndarray) -> float:
    """Indice de Davies-Bouldin via ``sklearn.metrics.davies_bouldin_score``.

    Le bruit DBSCAN (``-1``) est exclu avant le calcul : il ne constitue pas un
    cluster au sens de l'indice. Une valeur faible est meilleure (0 théorique).
    """
    labels = np.asarray(labels)
    valid = labels != -1
    X, y = matrix[valid], labels[valid]
    if len(np.unique(y)) < 2:
        return float("nan")
    return round(float(davies_bouldin_score(X, y)), 4)


def dunn_index(matrix: np.ndarray, labels: np.ndarray, sample_size: int = 2000) -> float:
    """Indice de Dunn : distance inter-clusters minimale / diamètre intra maximal.

    Implémentation locale, sans dépendance supplémentaire. L'échantillonnage est
    déterministe (graine ``RANDOM_STATE``) afin de rester reproductible sur les
    10 000 lignes. Une valeur élevée est meilleure.
    """
    labels = np.asarray(labels)
    valid = labels != -1
    X, y = matrix[valid], labels[valid]
    if len(X) > sample_size:
        rng = np.random.default_rng(RANDOM_STATE)
        selected = rng.choice(len(X), size=sample_size, replace=False)
        X, y = X[selected], y[selected]

    unique = np.unique(y)
    if len(unique) < 2:
        return float("nan")

    distances = squareform(pdist(X, metric="euclidean"))
    member_indices = [np.flatnonzero(y == cluster) for cluster in unique]
    diameters = [
        float(distances[np.ix_(idx, idx)].max()) if len(idx) > 1 else 0.0
        for idx in member_indices
    ]
    max_diameter = max(diameters)
    if max_diameter <= 0:
        return float("nan")

    min_separation = np.inf
    for i in range(len(member_indices)):
        for j in range(i + 1, len(member_indices)):
            block = distances[np.ix_(member_indices[i], member_indices[j])]
            if block.size:
                min_separation = min(min_separation, float(block.min()))
    if not np.isfinite(min_separation):
        return float("nan")
    return round(float(min_separation / max_diameter), 4)


def choose_k(
    matrix: np.ndarray, k_min: int = 2, k_max: int = 10
) -> tuple[pd.DataFrame, int]:
    """Compare k = 2..10 et retourne ``(tableau, k_retenu)``.

    Le tableau contient l'inertie, la silhouette, Davies-Bouldin et
    Calinski-Harabasz pour chaque k. Le k retenu est celui de la **meilleure
    silhouette**, à condition qu'il ne précède pas le coude de l'inertie : le
    coude est le k où la baisse relative d'inertie est maximale. Si la meilleure
    silhouette arrive avant ce coude, on garde le coude, ce qui évite de
    récompenser une partition trop grossière. Les métriques sont calculées sur
    la matrice encodée complète (silhouette échantillonnée à 2 000 lignes).
    """
    if k_min < 2:
        raise ValueError("k_min doit être au moins 2")
    k_max = min(int(k_max), len(matrix) - 1)
    if k_max < k_min:
        raise ValueError("k_max doit être supérieur ou égal à k_min et au nombre de lignes - 1")

    rows: list[dict[str, Any]] = []
    for k in range(k_min, k_max + 1):
        model = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
        labels = model.fit_predict(matrix)
        rows.append({
            "k": int(k),
            "inertie": round(float(model.inertia_), 2),
            "silhouette": _silhouette(matrix, labels),
            "davies_bouldin": davies_bouldin(matrix, labels),
            "calinski_harabasz": round(float(calinski_harabasz_score(matrix, labels)), 4),
        })

    table = pd.DataFrame(rows).set_index("k")
    inertias = table["inertie"].to_numpy(dtype=float)
    gains = np.zeros(len(table), dtype=float)
    if len(inertias) > 1:
        gains[1:] = (inertias[:-1] - inertias[1:]) / np.maximum(
            np.abs(inertias[:-1]), np.finfo(float).eps
        )
    elbow_k = int(table.index[int(np.argmax(gains))])
    silhouette_column = table["silhouette"].astype(float).fillna(-np.inf)
    best_silhouette_k = int(silhouette_column.idxmax())
    chosen_k = best_silhouette_k if best_silhouette_k >= elbow_k else elbow_k
    return table, chosen_k


def run_kmedoids(
    matrix: np.ndarray, raw: pd.DataFrame, n_clusters: int = DEFAULT_N_CLUSTERS,
    max_iter: int = 10,
) -> dict[str, Any]:
    """PAM simplifié : les médoïdes sont des lignes réelles de ``matrix``.

    Les candidats au rôle de médoïde sont un échantillon déterministe de
    ``MEDOID_CANDIDATE_SAMPLE`` points (au moins ``n_clusters``). L'algorithme
    alterne affectation au médoïde le plus proche et échange de médoïdes, en
    minimisant la somme des distances intra-cluster. Aucune dépendance
    supplémentaire n'est nécessaire.
    """
    n_rows = len(matrix)
    if not 1 <= n_clusters <= n_rows:
        raise ValueError("n_clusters doit être compris entre 1 et le nombre de lignes")

    rng = np.random.default_rng(RANDOM_STATE)
    n_candidates = min(n_rows, max(n_clusters, MEDOID_CANDIDATE_SAMPLE))
    candidate_positions = np.sort(rng.choice(n_rows, size=n_candidates, replace=False))
    distances = cdist(matrix, matrix[candidate_positions], metric="euclidean")
    medoid_positions = rng.choice(n_candidates, size=n_clusters, replace=False)

    for _ in range(max_iter):
        current = distances[:, medoid_positions]
        assignments = np.argmin(current, axis=1)
        current_cost = float(current[np.arange(n_rows), assignments].sum())
        best_swap: tuple[float, int, int] | None = None
        for slot in range(n_clusters):
            other_min = distances[:, np.delete(medoid_positions, slot)].min(axis=1)
            for candidate in range(n_candidates):
                if candidate in medoid_positions:
                    continue
                cost = float(np.minimum(other_min, distances[:, candidate]).sum())
                if cost < current_cost - 1e-9 and (best_swap is None or cost < best_swap[0]):
                    best_swap = (cost, slot, candidate)
        if best_swap is None:
            break
        medoid_positions[best_swap[1]] = best_swap[2]

    medoid_indices = candidate_positions[medoid_positions]
    labels = np.argmin(distances[:, medoid_positions], axis=1)
    result = _cluster_summary(labels, raw, matrix)
    result.update({
        "model": None,
        "labels": labels,
        "medoid_indices": [int(index) for index in medoid_indices],
        "medoids": matrix[medoid_indices].copy(),
        "n_candidates": int(n_candidates),
    })
    return result


def _max_diameter(matrix: np.ndarray, indices: np.ndarray) -> float:
    """Diamètre d'un cluster, calculé sur un échantillon déterministe si besoin."""
    selected = np.asarray(indices, dtype=int)
    if len(selected) > SILHOUETTE_SAMPLE_SIZE:
        rng = np.random.default_rng(RANDOM_STATE)
        selected = rng.choice(selected, size=SILHOUETTE_SAMPLE_SIZE, replace=False)
    if len(selected) < 2:
        return 0.0
    condensed = pdist(matrix[selected], metric="euclidean")
    return float(condensed.max()) if condensed.size else 0.0


def run_diana(matrix: np.ndarray, raw: pd.DataFrame, n_clusters: int = DEFAULT_N_CLUSTERS) -> dict[str, Any]:
    """DIANA simplifié : clustering divisif par scission du plus large diamètre.

    On part d'un seul groupe. À chaque itération, le cluster au diamètre maximal
    est scindé en deux : le point le plus éloigné du centroïde sert de graine,
    puis les points les plus éloignés de cette graine forment le nouveau sous-
    groupe (environ 30 % des points, au moins un). Les étiquettes finales sont
    déterministes, classées par plus petit index d'origine.
    """
    n_rows = len(matrix)
    if not 1 <= n_clusters <= n_rows:
        raise ValueError("n_clusters doit être compris entre 1 et le nombre de lignes")

    clusters: list[np.ndarray] = [np.arange(n_rows)]
    while len(clusters) < n_clusters:
        diameters = [_max_diameter(matrix, cluster) for cluster in clusters]
        splittable = [i for i, cluster in enumerate(clusters) if len(cluster) > 1]
        if not splittable:
            break
        target = max(splittable, key=lambda i: diameters[i])
        group = clusters[target]
        centroid = matrix[group].mean(axis=0)
        seed_index = int(np.argmax(np.linalg.norm(matrix[group] - centroid, axis=1)))
        distance_to_seed = np.linalg.norm(matrix[group] - matrix[group[seed_index]], axis=1)
        ordered = group[np.argsort(distance_to_seed)[::-1]]
        n_splinter = min(len(ordered) - 1, max(1, int(round(0.3 * len(ordered)))))
        splinter = np.sort(ordered[:n_splinter])
        remainder = np.sort(ordered[n_splinter:])
        clusters.pop(target)
        clusters.extend([remainder, splinter])

    clusters.sort(key=lambda cluster: int(cluster[0]))
    labels = np.empty(n_rows, dtype=int)
    for label, cluster in enumerate(clusters):
        labels[cluster] = label

    result = _cluster_summary(labels, raw, matrix)
    result.update({"model": None, "labels": labels, "n_groups": int(len(clusters))})
    return result


def run_cosine_kmeans(
    matrix: np.ndarray, raw: pd.DataFrame, n_clusters: int = DEFAULT_N_CLUSTERS
) -> dict[str, Any]:
    """K-Means sur vecteurs L2-normalisés (distance cosine équivalente)."""
    normalized = normalize(matrix, norm="l2")
    model = KMeans(n_clusters=n_clusters, n_init=10, random_state=RANDOM_STATE)
    labels = model.fit_predict(normalized)
    result = _cluster_summary(labels, raw, normalized)
    result.update({"model": model, "labels": labels, "metric": "cosine"})
    return result
