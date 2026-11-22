"""Analyses d'optimisation : défaut vs optimisé, sélection de variables, validation imbriquée.

Usage (depuis la racine du dépôt) :
    python scripts/run_optimization.py              # reprend là où le calcul s'est arrêté
    python scripts/run_optimization.py --skip-nested  # sans la validation imbriquée (~20 min)

Prérequis : python scripts/train_models.py (recherches en cache dans models/cache/).
Sorties : results/optimization/*.csv
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models import fit_search, get_searches  # noqa: E402
from src.optimization import (  # noqa: E402
    OPT_DIR, default_vs_tuned, feature_selection_study, random_search_simulation, run_nested,
)

PROCESSED = ROOT / "data" / "processed"


def main(skip_nested: bool = False):
    X_tr = pd.read_csv(PROCESSED / "X_train_raw.csv"); X_te = pd.read_csv(PROCESSED / "X_test_raw.csv")
    y_tr = pd.read_csv(PROCESSED / "y_train.csv").squeeze("columns"); y_te = pd.read_csv(PROCESSED / "y_test.csv").squeeze("columns")
    OPT_DIR.mkdir(parents=True, exist_ok=True)

    templates = get_searches()
    fitted = {n: fit_search(n, s, X_tr, y_tr) for n, s in templates.items()}      # cache models/cache

    t = time.time()
    if not (OPT_DIR / "defaut_vs_optimise.csv").exists():
        print("1/4 hyperparamètres par défaut vs optimisés", flush=True)
        default_vs_tuned(fitted, X_tr, y_tr, X_te, y_te).to_csv(OPT_DIR / "defaut_vs_optimise.csv")
    random_search_simulation(fitted).to_csv(OPT_DIR / "grid_vs_random.csv", index=False)
    print(f"   ok ({time.time() - t:.0f}s)", flush=True)

    if not (OPT_DIR / "selection_variables_resume.csv").exists():
        print("2/4 sélection de variables dans la validation croisée", flush=True)
        curve, summ = feature_selection_study(fitted, X_tr, y_tr, X_te, y_te)
        curve.to_csv(OPT_DIR / "selection_variables_courbe.csv", index=False)
        summ.to_csv(OPT_DIR / "selection_variables_resume.csv")

    if not skip_nested:
        print("3/4 validation croisée imbriquée (long)", flush=True)
        order = ["Naive Bayes (Gaussian)", "Naive Bayes (Bernoulli)", "Régression logistique", "Arbre de décision",
                 "K-NN", "Random Forest", "Gradient Boosting", "SVM"]
        run_nested(templates, fitted, X_tr, y_tr, OPT_DIR / "validation_imbriquee.csv", names=order)
    print("terminé", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--skip-nested", action="store_true")
    main(ap.parse_args().skip_nested)
