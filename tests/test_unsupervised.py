"""Contrats des modèles non supervisés — PR dédiée après le lab 1."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocessing import ALL_FEATURES, ID_COL, TARGET  # noqa: E402
from src.unsupervised import (  # noqa: E402
    UNSUPERVISED_MODELS,
    analyse_unsupervised,
    build_unsupervised_preprocessor,
    encode_features,
    run_agglomerative,
    run_dbscan,
    run_kmeans,
    run_pca,
)


def _sample_data() -> pd.DataFrame:
    """Petit jeu déterministe avec les 12 features du guide."""
    rows = []
    for index in range(12):
        rows.append({
            "Age": 25 + index,
            "JobLevel": 1 + (index % 3),
            "YearsAtCompany": index % 6,
            "MonthlyIncome": 3000 + index * 250,
            "JobSatisfaction": 1 + (index % 4),
            "WorkLifeBalance": 1 + (index % 4),
            "DistanceFromHome": 2 + index,
            "PerformanceRating": 3 + (index % 2),
            "TrainingHoursLastYear": index % 8,
            "Department": ["IT", "Sales", "HR"][index % 3],
            "OverTime": ["Yes", "No"][index % 2],
            "PromotionLast5Years": ["No", "Yes"][index % 2],
            ID_COL: f"EMP{index:04d}",
            TARGET: ["No", "Yes"][index % 2],
        })
    return pd.DataFrame(rows)


def test_registry_has_four_models():
    assert set(UNSUPERVISED_MODELS) == {
        "KMeans", "PCA", "AgglomerativeClustering", "DBSCAN"
    }


def test_preprocessor_contract():
    preprocessor = build_unsupervised_preprocessor()
    assert [name for name, _, _ in preprocessor.transformers] == ["num", "cat"]
    encoder = next(t for name, t, _ in preprocessor.transformers if name == "cat")
    assert encoder.handle_unknown == "ignore"


def test_encode_features_never_uses_id_or_target():
    df = _sample_data()
    matrix, names, raw = encode_features(df)
    assert matrix.shape[0] == len(df)
    assert matrix.shape[1] == len(names)
    assert raw[ALL_FEATURES].shape[1] == 12
    assert ID_COL not in raw[ALL_FEATURES].columns
    assert TARGET not in raw[ALL_FEATURES].columns
    assert np.isfinite(matrix).all()


def test_pca_returns_two_dimensions():
    matrix, names, _ = encode_features(_sample_data())
    result = run_pca(matrix, names, n_components=2)
    assert result["coordinates"].shape == (12, 2)
    assert 0.0 <= result["total_explained_variance"] <= 1.0
    assert len(result["components"]) == 2


def test_kmeans_and_agglomerative_labels():
    df = _sample_data()
    matrix, _, _ = encode_features(df)
    kmeans = run_kmeans(matrix, df, n_clusters=3)
    agglo = run_agglomerative(matrix, df, n_clusters=3)
    for result in (kmeans, agglo):
        labels = result["labels"]
        assert len(labels) == len(df)
        assert set(np.unique(labels)) <= {0, 1, 2}
    assert kmeans["inertia"] >= 0.0


def test_dbscan_labels_are_valid():
    df = _sample_data()
    matrix, names, _ = encode_features(df)
    coordinates = run_pca(matrix, names)["coordinates"]
    result = run_dbscan(coordinates, df, eps=1.0, min_samples=2)
    labels = result["labels"]
    assert len(labels) == len(df)
    assert set(np.unique(labels)).issubset(set(range(12)) | {-1})


def test_analysis_summary_is_json_serialisable():
    summary, details = analyse_unsupervised(_sample_data(), n_clusters=2, eps=1.0, min_samples=2)
    json.dumps(summary)
    assert summary["dataset"]["excluded_columns"] == [ID_COL, TARGET]
    assert summary["preprocessing"]["target_used_for_fitting"] is False
    assert set(summary["models"]) == set(UNSUPERVISED_MODELS)
    for labels in details["labels"].values():
        assert len(labels) == 12
