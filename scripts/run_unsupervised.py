"""Lance les modèles non supervisés et sauvegarde leur résumé.

Usage :
    python scripts/run_unsupervised.py
    python scripts/run_unsupervised.py --n-clusters 5 --eps 0.9 --min-samples 20

Le fichier généré est ``results/unsupervised_summary.json``. Il est volontairement
un artefact d'analyse : il doit être ignoré par Git au même titre que les autres
fichiers de ``results/``.
"""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Console Windows (cp1252) : éviter UnicodeEncodeError sur les messages JSON/UI.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

from src.unsupervised import (  # noqa: E402
    DEFAULT_DBSCAN_EPS,
    DEFAULT_DBSCAN_MIN_SAMPLES,
    DEFAULT_N_CLUSTERS,
    analyse_unsupervised,
)

RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT = RESULTS_DIR / "unsupervised_summary.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse non supervisée de l'attrition")
    parser.add_argument("--n-clusters", type=int, default=DEFAULT_N_CLUSTERS)
    parser.add_argument("--eps", type=float, default=DEFAULT_DBSCAN_EPS)
    parser.add_argument("--min-samples", type=int, default=DEFAULT_DBSCAN_MIN_SAMPLES)
    args = parser.parse_args()

    summary, _ = analyse_unsupervised(
        n_clusters=args.n_clusters,
        eps=args.eps,
        min_samples=args.min_samples,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(f"✔ Sauvegardé : {OUTPUT.relative_to(PROJECT_ROOT)}")

    print("\n" + "=" * 64)
    print("  RÉSUMÉ — MODÈLES NON SUPERVISÉS")
    print("=" * 64)
    dataset = summary["dataset"]
    print(f"  Lignes : {dataset['rows']}  |  Features sources : {dataset['input_features']}"
          f"  |  Features encodées : {dataset['encoded_features']}")
    print(f"  Colonnes exclues : {', '.join(dataset['excluded_columns'])}")

    pca = summary["PCA"]
    print(f"\n  PCA        : variance expliquée totale = {pca['total_explained_variance']}")
    for component in pca["components"]:
        top = component["top_features"][0]
        print(f"    PC{component['component']} : {component['explained_variance_ratio']}"
              f" — {top['feature']} ({top['loading']})")

    for name in ("KMeans", "AgglomerativeClustering", "DBSCAN"):
        result = summary[name]
        silhouette = result.get("silhouette")
        silhouette_text = "N/A" if silhouette is None else f"{silhouette}"
        extra = ""
        if name == "KMeans":
            extra = f"  |  inertie = {result['inertia']}"
        if name == "DBSCAN":
            extra = f"  |  bruit = {result.get('n_noise', 0)}"
        print(f"  {name:<24} clusters = {result['n_clusters']}  |  silhouette = {silhouette_text}{extra}")

    print("\n  Remarque : les taux d'attrition par cluster sont descriptifs, jamais utilisés pour l'ajustement.")
    print("=" * 64)


if __name__ == "__main__":
    main()
