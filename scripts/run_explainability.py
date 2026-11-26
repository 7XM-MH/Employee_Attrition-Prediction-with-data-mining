"""Explicabilité locale reproductible : top-3 SHAP par employé.

Usage : python scripts/run_explainability.py

Produit :
    results/explainability_top3.csv
    reports/figures/42_shap_local.png

Le modèle utilisé est un arbre peu profond, déterministe, entraîné sur le jeu
d'apprentissage avec le préprocesseur du guide. Aucun artefact joblib n'est écrit.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.explainability import explain_many
from src.preprocessing import (
    FEATURE_COLS,
    ID_COL,
    TARGET,
    build_preprocessor,
    clean_data,
    load_raw,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "Employee_Attrition_DataSet.csv"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "reports" / "figures"
RANDOM_STATE = 42
SAMPLE_SIZE = 200


def build_bundle(X: pd.DataFrame, y: pd.Series) -> dict:
    pipe = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", DecisionTreeClassifier(
            max_depth=4, class_weight="balanced", random_state=RANDOM_STATE
        )),
    ]).fit(X, y)
    return {"name": "DecisionTree", "model": pipe, "threshold": 0.5, "features": FEATURE_COLS}


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df, report = clean_data(load_raw(DATA_PATH))
    X, y = df[FEATURE_COLS], df[TARGET]
    X_train, X_test, y_train, _ = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    bundle = build_bundle(X_train, y_train)

    rng = np.random.default_rng(RANDOM_STATE)
    size = min(SAMPLE_SIZE, len(X_test))
    idx = rng.choice(len(X_test), size=size, replace=False)
    sample = X_test.iloc[idx].reset_index(drop=True)

    table = explain_many(bundle, sample, top_k=3)
    csv_path = RESULTS_DIR / "explainability_top3.csv"
    table.to_csv(csv_path, index=False)
    print(f"  ✔ {csv_path.relative_to(ROOT)} : {len(table)} facteurs expliqués")

    importance = table.assign(abs_contribution=table["contribution"].abs())
    means = (
        importance.groupby("feature")["abs_contribution"]
        .mean()
        .sort_values(ascending=True)
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(means.index, means.values, color="#2c7fb8")
    ax.set_xlabel("Contribution SHAP moyenne (valeur absolue)")
    ax.set_title("Explicabilité locale — top 3 par employé (200 employés)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig_path = FIGURES_DIR / "42_shap_local.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"  ✔ {fig_path.relative_to(ROOT)}")

    print(f"\n  Lignes nettoyées : {report['rows_clean']} | "
          f"exclus de X : {ID_COL}, {TARGET}")
    print("\n  Exemple de prédiction expliquée :")
    print(table.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
