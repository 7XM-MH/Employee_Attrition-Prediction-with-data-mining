"""Clustering avancé (lot 3) : choix de k, PAM, DIANA, cosinus et indices.

Usage depuis la racine du dépôt :
    python scripts/run_unsupervised_advanced.py
    python scripts/run_unsupervised_advanced.py --k-min 2 --k-max 10 --k 4

Sorties versionnées :
    results/clustering_advanced.csv
    reports/figures/37_choix_k.png ... 41_clusters_cosinus.png

``Attrition`` et ``EmployeeID`` ne sont jamais utilisés pour ajuster les
modèles : la cible ne sert qu'à décrire les groupes après coup, dans la colonne
``attrition_rate_by_cluster`` du CSV.
"""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Console Windows (cp1252) : éviter UnicodeEncodeError sur les messages accentués.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.cluster.hierarchy import dendrogram, linkage  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402

from src.preprocessing import ID_COL, TARGET, load_raw  # noqa: E402
from src.unsupervised import (  # noqa: E402
    DENDROGRAM_SAMPLE_SIZE,
    DEFAULT_DBSCAN_EPS,
    DEFAULT_DBSCAN_MIN_SAMPLES,
    DEFAULT_DATA_PATH,
    DEFAULT_N_CLUSTERS,
    RANDOM_STATE,
    _silhouette,
    choose_k,
    davies_bouldin,
    dunn_index,
    encode_features,
    run_agglomerative,
    run_cosine_kmeans,
    run_dbscan,
    run_diana,
    run_kmedoids,
    run_kmeans,
    run_pca,
)

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
OUTPUT = RESULTS_DIR / "clustering_advanced.csv"
COLUMNS = [
    "method", "k", "inertie", "silhouette", "davies_bouldin", "dunn", "n_clusters",
    "noise", "cluster_sizes", "attrition_rate_by_cluster", "note",
]


def _json_mapping(mapping: dict) -> str:
    return json.dumps(mapping, ensure_ascii=False, sort_keys=True)


def _attrition_rates(labels: np.ndarray, raw: pd.DataFrame) -> str:
    """Taux de départ par cluster, calculé après l'ajustement uniquement."""
    if TARGET not in raw.columns:
        return ""
    valid = labels != -1
    target = raw[TARGET].astype(str).to_numpy()
    rates = (
        pd.DataFrame({"cluster": labels[valid], TARGET: target[valid]})
        .groupby("cluster")[TARGET]
        .apply(lambda values: (values == "Yes").mean())
    )
    return _json_mapping({str(int(k)): round(float(v), 4) for k, v in rates.items()})


def _metrics(matrix: np.ndarray, labels: np.ndarray) -> dict:
    labels = np.asarray(labels)
    valid = labels != -1
    counts = pd.Series(labels[valid]).value_counts().sort_index()
    return {
        "silhouette": _silhouette(matrix, labels),
        "davies_bouldin": davies_bouldin(matrix, labels),
        "dunn": dunn_index(matrix, labels),
        "n_clusters": int(counts.size),
        "noise": int((~valid).sum()),
        "cluster_sizes": _json_mapping({str(int(k)): int(v) for k, v in counts.items()}),
    }


def _row(method: str, k: int, result: dict, note: str = "") -> dict:
    row = {"method": method, "k": int(k)}
    for column in ("inertie", "silhouette", "davies_bouldin", "dunn", "n_clusters", "noise", "cluster_sizes"):
        row[column] = result.get(column)
    row["attrition_rate_by_cluster"] = result.get("attrition_rate_by_cluster", "")
    row["note"] = note
    return row


def _save(fig, filename: str) -> None:
    target = FIGURES_DIR / filename
    fig.savefig(target, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure : {target.relative_to(PROJECT_ROOT)}")


def _figure_choix_k(table: pd.DataFrame, chosen_k: int) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for axis, column, title in zip(
        axes,
        ("inertie", "silhouette", "davies_bouldin"),
        ("Inertie (minuscule = mieux)", "Silhouette (élevé = mieux)", "Davies-Bouldin (faible = mieux)"),
    ):
        axis.plot(table.index, table[column].astype(float), marker="o", color="#2563a6")
        axis.axvline(chosen_k, color="#c0392b", linestyle="--", linewidth=1.4, label=f"k retenu = {chosen_k}")
        axis.set_title(title)
        axis.set_xlabel("k")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle("Choix du nombre de groupes — K-Means (graine fixe)", fontsize=13)
    fig.tight_layout()
    _save(fig, "37_choix_k.png")


def _figure_dendrogram(matrix: np.ndarray) -> None:
    rng = np.random.default_rng(RANDOM_STATE)
    size = min(DENDROGRAM_SAMPLE_SIZE, len(matrix))
    sample = matrix[rng.choice(len(matrix), size=size, replace=False)]
    linked = linkage(sample, method="ward")
    fig, axis = plt.subplots(figsize=(13, 5))
    dendrogram(
        linked, truncate_mode="lastp", p=30, no_labels=True,
        color_threshold=0.7 * float(linked[:, 2].max()), ax=axis,
    )
    axis.set_title(
        f"Dendrogramme de Ward — échantillon déterministe de {size} lignes "
        f"sur {len(matrix):,}"
    )
    axis.set_xlabel("Groupes fusionnés (30 derniers niveaux)")
    axis.set_ylabel("Distance de fusion")
    fig.tight_layout()
    _save(fig, "38_dendrogramme.png")


def _figure_kmedoids_diana(coordinates: np.ndarray, kmedoids: dict, diana: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, result, title in (
        (axes[0], kmedoids, "K-Medoids (PAM simplifié)"),
        (axes[1], diana, "DIANA (divisif)"),
    ):
        scatter = axis.scatter(
            coordinates[:, 0], coordinates[:, 1], c=result["labels"],
            cmap="tab10", s=9, alpha=0.7,
        )
        axis.set_title(title)
        axis.set_xlabel("PC1")
        axis.set_ylabel("PC2")
        fig.colorbar(scatter, ax=axis, shrink=0.8)
    if "medoid_indices" in kmedoids:
        indices = kmedoids["medoid_indices"]
        axes[0].scatter(
            coordinates[indices, 0], coordinates[indices, 1],
            marker="X", s=130, c="red", edgecolors="black", label="médoïdes",
        )
        axes[0].legend(loc="best")
    fig.suptitle("Projections PCA des deux nouvelles partitions", fontsize=13)
    fig.tight_layout()
    _save(fig, "39_kmedoids_diana.png")


def _figure_indices(rows: list[dict], methods: list[str]) -> None:
    selected = [row for row in rows if row["method"] in methods]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for axis, column, title in zip(
        axes,
        ("silhouette", "davies_bouldin", "dunn"),
        ("Silhouette (élevé = mieux)", "Davies-Bouldin (faible = mieux)", "Dunn (élevé = mieux)"),
    ):
        values = [
            float(row[column]) if row[column] is not None and np.isfinite(row[column]) else 0.0
            for row in selected
        ]
        axis.barh([row["method"] for row in selected], values, color="#3b7ea1")
        axis.set_title(title)
        axis.grid(axis="x", alpha=0.25)
    fig.suptitle("Indices de validation au k retenu", fontsize=13)
    fig.tight_layout()
    _save(fig, "40_indices_clustering.png")


def _figure_cosinus(coordinates: np.ndarray, cosine: dict) -> None:
    fig, axis = plt.subplots(figsize=(7.5, 5.5))
    scatter = axis.scatter(
        coordinates[:, 0], coordinates[:, 1], c=cosine["labels"],
        cmap="tab10", s=9, alpha=0.7,
    )
    axis.set_title("K-Means en similarité cosinus (matrice L2-normalisée)")
    axis.set_xlabel("PC1")
    axis.set_ylabel("PC2")
    fig.colorbar(scatter, ax=axis, shrink=0.8)
    fig.tight_layout()
    _save(fig, "41_clusters_cosinus.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Clustering avancé du mini-projet Attrition")
    parser.add_argument("--input", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=10)
    parser.add_argument("--k", type=int, default=None, help="force k au lieu du meilleur choix")
    parser.add_argument("--eps", type=float, default=DEFAULT_DBSCAN_EPS)
    parser.add_argument("--min-samples", type=int, default=DEFAULT_DBSCAN_MIN_SAMPLES)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_raw(args.input)
    matrix, feature_names, raw = encode_features(raw)
    print(f"Matrice encodée : {matrix.shape[0]:,} lignes x {matrix.shape[1]} colonnes")
    print(f"Colonnes exclues : {ID_COL}, {TARGET}")

    table, suggested_k = choose_k(matrix, args.k_min, args.k_max)
    n_clusters = int(args.k) if args.k is not None else suggested_k
    print(f"k suggéré par la silhouette et le coude : {suggested_k} | k utilisé : {n_clusters}")

    pca = run_pca(matrix, feature_names)
    kmeans = run_kmeans(matrix, raw, n_clusters)
    agglomerative = run_agglomerative(matrix, raw, n_clusters)
    dbscan = run_dbscan(pca["coordinates"], raw, args.eps, args.min_samples)
    kmedoids = run_kmedoids(matrix, raw, n_clusters)
    diana = run_diana(matrix, raw, n_clusters)
    cosine = run_cosine_kmeans(matrix, raw, n_clusters)

    rows: list[dict] = []
    # Balayage de k : toutes les métriques sont recalculées sur les mêmes labels
    # que ceux utilisés pour le choix, avec la même graine.
    for k in table.index:
        labels = KMeans(n_clusters=int(k), n_init=10, random_state=RANDOM_STATE).fit_predict(matrix)
        metrics = _metrics(matrix, labels)
        metrics["inertie"] = float(table.loc[k, "inertie"])
        metrics["attrition_rate_by_cluster"] = _attrition_rates(labels, raw)
        rows.append(_row("KMeans", int(k), metrics, "balayage du choix de k"))

    methods_at_k = [
        ("AgglomerativeClustering", agglomerative, "Ward + connectivité 10-PPV"),
        ("DBSCAN", dbscan, f"eps={args.eps}, min_samples={args.min_samples} ; k indicatif"),
        ("KMedoids", kmedoids, "PAM simplifié, médoïdes = points réels"),
        ("DIANA", diana, "clustering divisif, scission du plus grand diamètre"),
        ("CosineKMeans", cosine, "K-Means sur matrice L2-normalisée"),
    ]
    selected_rows: list[dict] = []
    for name, result, note in methods_at_k:
        metrics = _metrics(matrix, result["labels"])
        metrics["attrition_rate_by_cluster"] = _attrition_rates(result["labels"], raw)
        row = _row(name, n_clusters, metrics, note)
        rows.append(row)
        selected_rows.append(row)

    kmeans_metrics = _metrics(matrix, kmeans["labels"])
    kmeans_metrics["inertie"] = kmeans["inertia"]
    kmeans_metrics["attrition_rate_by_cluster"] = _attrition_rates(kmeans["labels"], raw)
    kmeans_row = _row("KMeans", n_clusters, kmeans_metrics, "k retenu")
    rows.append(kmeans_row)
    selected_rows.append(kmeans_row)

    rows.append({
        "method": "PCA", "k": 2, "inertie": None, "silhouette": None, "davies_bouldin": None,
        "dunn": None, "n_clusters": None, "noise": None, "cluster_sizes": "",
        "attrition_rate_by_cluster": "", "note": "projection 2D, pas de partition",
    })

    output = pd.DataFrame(rows, columns=COLUMNS)
    output.to_csv(OUTPUT, index=False)
    print(f"✔ Sauvegardé : {OUTPUT.relative_to(PROJECT_ROOT)}")

    _figure_choix_k(table, suggested_k)
    _figure_dendrogram(matrix)
    _figure_kmedoids_diana(pca["coordinates"], kmedoids, diana)
    _figure_indices(
        selected_rows,
        ["KMeans", "AgglomerativeClustering", "DBSCAN", "KMedoids", "DIANA", "CosineKMeans"],
    )
    _figure_cosinus(pca["coordinates"], cosine)

    print("\n" + "=" * 72)
    print(f"  k retenu : {n_clusters}")
    print(f"  {'méthode':<24} {'silhouette':>10} {'Davies-Bouldin':>15} {'Dunn':>8}")
    for row in selected_rows:
        print(f"  {row['method']:<24} {str(row['silhouette']):>10} "
              f"{str(row['davies_bouldin']):>15} {str(row['dunn']):>8}")
    print("  Taux d'attrition par cluster : descriptifs, calculés après l'ajustement.")
    print("=" * 72)


if __name__ == "__main__":
    main()
