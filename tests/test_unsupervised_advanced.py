"""Tests déterministes du clustering avancé (lot 3).

Les jeux synthétiques sont petits et à trois blocs bien séparés. Aucune colonne
cible n'est utilisée par l'encodeur : un jeu avec une colonne ``cible`` modifiée
doit produire exactement la même matrice et les mêmes étiquettes.
"""
import numpy as np
import pandas as pd

from src.preprocessing import ID_COL, TARGET
from src.unsupervised import (
    UNSUPERVISED_MODELS,
    choose_k,
    davies_bouldin,
    dunn_index,
    encode_features,
    run_cosine_kmeans,
    run_diana,
    run_kmedoids,
    run_kmeans,
)

N_PER_CLUSTER = 40
RANDOM_STATE = 11


def _make_frame() -> pd.DataFrame:
    """Trois groupes synthétiques très séparés, avec une colonne « cible » piège."""
    rng = np.random.default_rng(RANDOM_STATE)
    frames = []
    for cluster in range(3):
        size = N_PER_CLUSTER
        frames.append(pd.DataFrame({
            "Age": 25 + 15 * cluster + rng.normal(0, 0.4, size),
            "YearsAtCompany": 1 + 7 * cluster + rng.normal(0, 0.3, size),
            "MonthlyIncome": 3500 + 5500 * cluster + rng.normal(0, 40, size),
            "DistanceFromHome": 2 + 14 * cluster + rng.normal(0, 0.4, size),
            "TrainingHoursLastYear": 4 + 7 * cluster + rng.normal(0, 0.3, size),
            "JobLevel": np.full(size, 1 + cluster),
            "JobSatisfaction": np.full(size, 1 + cluster),
            "WorkLifeBalance": np.full(size, 1 + cluster),
            "PerformanceRating": np.full(size, 1 + cluster),
            "OverTime": np.full(size, "No" if cluster != 1 else "Yes", dtype=object),
            "PromotionLast5Years": np.full(size, "No", dtype=object),
            "Department": np.full(size, ["Sales", "IT", "HR"][cluster], dtype=object),
            TARGET: np.full(size, "Yes" if cluster == 2 else "No", dtype=object),
            ID_COL: [f"EMP{cluster * size + i:04d}" for i in range(size)],
            "cible": np.full(size, "Yes" if cluster == 2 else "No", dtype=object),
        }))
    frame = pd.concat(frames, ignore_index=True)
    return frame.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)


def _matrix_and_frame():
    frame = _make_frame()
    matrix, _, raw = encode_features(frame)
    return matrix, raw


def test_advanced_models_registered():
    expected = {
        "KMeans", "PCA", "AgglomerativeClustering", "DBSCAN",
        "KMedoids", "DIANA", "CosineKMeans",
    }
    assert expected.issubset(UNSUPERVISED_MODELS)


def test_choose_k_covers_range_and_returns_valid_k():
    matrix, _ = _matrix_and_frame()

    table, chosen_k = choose_k(matrix, k_min=2, k_max=10)

    assert list(table.index) == list(range(2, 11))
    assert {"inertie", "silhouette", "davies_bouldin", "calinski_harabasz"}.issubset(table.columns)
    assert 2 <= chosen_k <= 10
    assert np.isfinite(table["inertie"].to_numpy(dtype=float)).all()


def test_kmedoids_labels_and_real_medoids():
    matrix, frame = _matrix_and_frame()

    result = run_kmedoids(matrix, frame, n_clusters=3)
    labels = np.asarray(result["labels"])

    assert len(np.unique(labels)) == 3
    assert set(np.unique(labels)).issubset({0, 1, 2})
    indices = np.asarray(result["medoid_indices"], dtype=int)
    assert len(indices) == 3
    assert indices.min() >= 0 and indices.max() < len(frame)
    np.testing.assert_array_equal(np.asarray(result["medoids"]), matrix[indices])
    for medoid_index, label in zip(indices, labels[indices]):
        assert labels[medoid_index] == label


def test_diana_returns_exact_number_of_clusters():
    matrix, frame = _matrix_and_frame()

    result = run_diana(matrix, frame, n_clusters=3)

    assert result["n_groups"] == 3
    assert len(np.unique(np.asarray(result["labels"]))) == 3


def test_validation_indices_are_finite_and_better_on_separated_blocks():
    matrix, frame = _matrix_and_frame()
    labels = run_kmeans(matrix, frame, n_clusters=3)["labels"]
    rng = np.random.default_rng(RANDOM_STATE)
    shuffled = matrix[rng.permutation(len(matrix))]

    db_separated = davies_bouldin(matrix, labels)
    db_shuffled = davies_bouldin(shuffled, labels)
    dunn_separated = dunn_index(matrix, labels, sample_size=len(matrix))
    dunn_shuffled = dunn_index(shuffled, labels, sample_size=len(matrix))

    for value in (db_separated, db_shuffled, dunn_separated, dunn_shuffled):
        assert np.isfinite(value)
    assert db_separated < db_shuffled
    assert dunn_separated > dunn_shuffled


def test_no_target_leakage_from_extra_column():
    matrix_a, frame_a = _matrix_and_frame()
    frame_b = frame_a.copy()
    frame_b["cible"] = np.where(frame_b["cible"] == "Yes", "No", "Yes")
    frame_b[TARGET] = np.where(frame_b[TARGET] == "Yes", "No", "Yes")
    frame_b[ID_COL] = "EMP9999"
    matrix_b, _, _ = encode_features(frame_b)

    np.testing.assert_array_equal(matrix_a, matrix_b)
    np.testing.assert_array_equal(
        run_kmedoids(matrix_a, frame_a, n_clusters=3)["labels"],
        run_kmedoids(matrix_b, frame_b, n_clusters=3)["labels"],
    )


def test_cosine_kmeans_labels_valid():
    matrix, frame = _matrix_and_frame()

    result = run_cosine_kmeans(matrix, frame, n_clusters=3)

    assert len(np.unique(np.asarray(result["labels"]))) == 3
    assert result["metric"] == "cosine"
