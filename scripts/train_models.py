"""Entraîne, règle (GridSearchCV / RandomizedSearchCV, K=5) et évalue tous les modèles.

Usage (depuis la racine du dépôt) :
    python scripts/train_models.py            # réutilise models/cache/ s'il existe
    python scripts/train_models.py --force    # relance toutes les recherches

Sorties : models/best_model.joblib, models/<modèle>.joblib,
          data/processed/model_comparison.csv, results/*.csv|json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation import bootstrap_ci, evaluate_models, fold_scores, friedman_test  # noqa: E402
from src.models import (  # noqa: E402
    BASELINE, MODELS_DIR, cv_summary, final_estimator, fit_search, get_searches,
    get_zero_r, save_bundle, slug,
)

PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"


def load_split():
    X_tr = pd.read_csv(PROCESSED / "X_train_raw.csv")
    X_te = pd.read_csv(PROCESSED / "X_test_raw.csv")
    y_tr = pd.read_csv(PROCESSED / "y_train.csv").squeeze("columns")
    y_te = pd.read_csv(PROCESSED / "y_test.csv").squeeze("columns")
    return X_tr, X_te, y_tr, y_te


def main(force: bool = False):
    t0 = time.time()
    X_tr, X_te, y_tr, y_te = load_split()
    RESULTS.mkdir(exist_ok=True)

    print("1/3 Recherche d'hyperparamètres (validation croisée 5 plis)")
    searches = {}
    for name, s in get_searches().items():
        t = time.time()
        searches[name] = fit_search(name, s, X_tr, y_tr, force=force)
        print(f"  {name:<26} AP_cv={searches[name].best_score_:.4f}  ({time.time() - t:.0f}s)", flush=True)

    summary = cv_summary(searches)
    summary.to_csv(RESULTS / "cv_summary.csv")
    fold_scores(searches).to_csv(RESULTS / "cv_fold_scores_ap.csv", index=False)

    print("2/3 Évaluation sur le jeu de test")
    estimators = {BASELINE: get_zero_r()}
    estimators.update({n: final_estimator(n, s) for n, s in searches.items()})
    ev = evaluate_models(estimators, X_tr, y_tr, X_te, y_te)
    table = ev["table"]

    ci = bootstrap_ci(y_te, ev["probs"])
    table = table.join(ci[["roc_auc_low", "roc_auc_high", "pr_auc_low", "pr_auc_high"]])
    table.to_csv(PROCESSED / "model_comparison.csv")
    table.to_csv(RESULTS / "model_comparison.csv")

    print("3/3 Sauvegarde des modèles")
    for name, model in ev["fitted"].items():
        joblib.dump(model, MODELS_DIR / f"{slug(name)}.joblib", compress=3)
    best = summary["AP_cv"].idxmax()         # choix sur la validation croisée, pas sur le test
    save_bundle(best, ev["fitted"][best], ev["thresholds"][best],
                info={"AP_cv": float(summary.loc[best, "AP_cv"]),
                      "test_roc_auc": float(table.loc[best, "roc_auc"]),
                      "test_f1_opt": float(table.loc[best, "f1_opt"])})
    fr = friedman_test(fold_scores(searches))
    (RESULTS / "friedman_test.json").write_text(json.dumps(
        {"statistique": fr["statistique"], "p_value": fr["p_value"],
         "rang_moyen": fr["rang_moyen"].to_dict()}, ensure_ascii=False, indent=2), encoding="utf-8")

    pd.options.display.width = 200
    print("\n", table[["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "seuil_opt", "f1_opt"]].round(3))
    print(f"\nMeilleur modèle (AP en CV) : {best}  | durée totale : {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    main(ap.parse_args().force)
