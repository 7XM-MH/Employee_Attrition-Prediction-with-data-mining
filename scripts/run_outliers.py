"""Génère le rapport descriptif des outliers (4 méthodes) et la figure 36.

Usage, depuis la racine du dépôt :

    python scripts/run_outliers.py

Le script lit ``data/raw/Employee_Attrition_DataSet.csv``, applique les quatre
méthodes de ``src.outliers`` aux variables numériques et ordinales, puis écrit
deux livrables **versionnés** :

* ``results/outliers_report.csv``
* ``reports/figures/36_outliers_methodes.png``

Analyse descriptive sur le jeu brut : ``Attrition`` et ``EmployeeID`` ne sont
jamais utilisés pour détecter les outliers. Aucun modèle de prédiction n'est
modifié.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Console Windows (cp1252) : éviter UnicodeEncodeError sur les messages accentués.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.outliers import outlier_summary  # noqa: E402
from src.preprocessing import load_raw  # noqa: E402

RAW_RELATIVE_PATH = Path("data/raw/Employee_Attrition_DataSet.csv")
CSV_RELATIVE_PATH = Path("results/outliers_report.csv")
FIGURE_RELATIVE_PATH = Path("reports/figures/36_outliers_methodes.png")

_METHOD_COLORS = {
    "Z-score": "#4C72B0",
    "IQR": "#DD8452",
    "Isolation Forest": "#55A868",
    "LOF": "#C44E52",
}


def _plot_summary(summary: pd.DataFrame, path: Path) -> None:
    """Trace la comparaison des quatre méthodes (pourcentage d'outliers)."""
    order = list(_METHOD_COLORS)
    plotted = summary.copy()
    plotted["_ordre"] = plotted["methode"].apply(
        lambda m: order.index(m) if m in order else len(order)
    )
    plotted = plotted.sort_values(["_ordre", "pct_outliers"], ascending=[True, False])
    labels = [
        f"{row.methode} — {row.variable}" for row in plotted.itertuples(index=False)
    ]
    colors = [_METHOD_COLORS.get(m, "#8172B3") for m in plotted["methode"]]

    fig, ax = plt.subplots(figsize=(10.5, 9))
    bars = ax.barh(range(len(plotted)), plotted["pct_outliers"], color=colors)
    ax.set_yticks(range(len(plotted)), labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Part des outliers détectés (%)")
    ax.set_title(
        "Détection des outliers — comparaison des 4 méthodes\n"
        "Jeu brut : analyse descriptive, sans Attrition ni EmployeeID"
    )
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, plotted["pct_outliers"]):
        ax.text(
            value + 0.02, bar.get_y() + bar.get_height() / 2,
            f"{value:.2f} %", va="center", fontsize=7,
        )

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=color) for color in _METHOD_COLORS.values()
    ]
    ax.legend(handles, order, loc="lower right", fontsize=8, title="Méthode")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(root: Path = PROJECT_ROOT) -> pd.DataFrame:
    """Exécute l'analyse depuis ``root`` et renvoie le DataFrame écrit en CSV."""
    root = Path(root)
    raw_path = root / RAW_RELATIVE_PATH
    if not raw_path.exists():
        raise FileNotFoundError(f"Jeu brut introuvable : {raw_path}")

    df = load_raw(raw_path)
    summary = outlier_summary(df)

    csv_path = root / CSV_RELATIVE_PATH
    figure_path = root / FIGURE_RELATIVE_PATH
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(csv_path, index=False)
    _plot_summary(summary, figure_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=PROJECT_ROOT,
        help="Racine du dépôt (par défaut : racine calculée depuis ce script).",
    )
    args = parser.parse_args()
    summary = run(args.root)
    print(f"Lignes du rapport : {len(summary)}")
    print(summary.to_string(index=False))
    print(f"CSV  : {args.root / CSV_RELATIVE_PATH}")
    print(f"Figure : {args.root / FIGURE_RELATIVE_PATH}")


if __name__ == "__main__":
    main()
