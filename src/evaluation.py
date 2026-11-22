"""Évaluation comparative des modèles.

Principe : le jeu de test n'est utilisé qu'à la toute fin. Le seuil de décision
de chaque modèle est choisi sur le **train** (probabilités hors-pli obtenues
par validation croisée), puis appliqué tel quel au test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss, confusion_matrix,
    f1_score, matthews_corrcoef, precision_recall_curve, precision_score, recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_predict

from src.models import BASELINE, CV


# --------------------------------------------------------------------------- #
# Métriques
# --------------------------------------------------------------------------- #
def metrics_at(y_true, p, threshold: float = 0.5) -> dict:
    """Métriques de classification à un seuil donné (classe positive = départ)."""
    pred = (np.asarray(p) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, pred),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def best_f1_threshold(y_true, p) -> tuple[float, float]:
    """Seuil qui maximise le F1 de la classe positive (courbe précision-rappel)."""
    prec, rec, thr = precision_recall_curve(y_true, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    i = int(np.argmax(f1))
    return float(thr[i]), float(f1[i])


# --------------------------------------------------------------------------- #
# Évaluation de tous les modèles
# --------------------------------------------------------------------------- #
def evaluate_models(estimators: dict, X_train, y_train, X_test, y_test, cv=CV, verbose=True) -> dict:
    """Ajuste chaque estimateur sur le train et l'évalue sur le test.

    Pour chaque modèle :
      1. probabilités hors-pli sur le train -> seuil optimal (F1) ;
      2. ajustement sur tout le train ;
      3. métriques sur le test au seuil 0,5 et au seuil optimisé.

    Renvoie un dict avec ``table``, ``probs`` (test), ``oof`` (train),
    ``thresholds`` et ``fitted``.
    """
    rows, probs, oofs, thresholds, fitted = [], {}, {}, {}, {}
    for name, est in estimators.items():
        if verbose:
            print(f"  évaluation : {name}", flush=True)
        if name == BASELINE:
            thr, oof = 0.5, None          # probabilité constante : seuil sans objet
        else:
            oof = cross_val_predict(clone(est), X_train, y_train, cv=cv, method="predict_proba")[:, 1]
            thr, _ = best_f1_threshold(y_train, oof)
        model = clone(est).fit(X_train, y_train)
        p = model.predict_proba(X_test)[:, 1]

        m05 = metrics_at(y_test, p, 0.5)
        mop = metrics_at(y_test, p, thr)
        rows.append({
            "modèle": name,
            "accuracy": m05["accuracy"], "precision": m05["precision"],
            "recall": m05["recall"], "f1": m05["f1"],
            "roc_auc": roc_auc_score(y_test, p), "pr_auc": average_precision_score(y_test, p),
            "brier": brier_score_loss(y_test, p),
            "seuil_opt": thr,
            "precision_opt": mop["precision"], "recall_opt": mop["recall"], "f1_opt": mop["f1"],
            "balanced_acc_opt": mop["balanced_accuracy"], "mcc_opt": mop["mcc"],
        })
        probs[name], oofs[name], thresholds[name], fitted[name] = p, oof, thr, model
    table = pd.DataFrame(rows).set_index("modèle")
    return {"table": table, "probs": probs, "oof": oofs, "thresholds": thresholds, "fitted": fitted}


# --------------------------------------------------------------------------- #
# Incertitude : intervalles de confiance par bootstrap (rééchantillonnage apparié)
# --------------------------------------------------------------------------- #
def bootstrap_ci(y_true, probs: dict, n_boot: int = 1000, seed: int = 42, alpha: float = 0.05) -> pd.DataFrame:
    """IC à 95 % du ROC-AUC et du PR-AUC sur le jeu de test.

    Les mêmes rééchantillons sont utilisés pour tous les modèles (comparaison
    appariée). Le jeu de test ne contenant que ~180 départs, l'incertitude est
    importante : des différences d'AUC inférieures à l'IC ne sont pas fiables.
    """
    y = np.asarray(y_true)
    rng = np.random.default_rng(seed)
    idx = [rng.integers(0, len(y), len(y)) for _ in range(n_boot)]
    idx = [i for i in idx if 0 < y[i].sum() < len(i)]
    rows = []
    for name, p in probs.items():
        p = np.asarray(p)
        auc = np.array([roc_auc_score(y[i], p[i]) for i in idx])
        ap = np.array([average_precision_score(y[i], p[i]) for i in idx])
        lo, hi = alpha / 2 * 100, (1 - alpha / 2) * 100
        rows.append({
            "modèle": name,
            "roc_auc": roc_auc_score(y, p), "roc_auc_low": np.percentile(auc, lo), "roc_auc_high": np.percentile(auc, hi),
            "pr_auc": average_precision_score(y, p), "pr_auc_low": np.percentile(ap, lo), "pr_auc_high": np.percentile(ap, hi),
        })
    return pd.DataFrame(rows).set_index("modèle")


def paired_bootstrap_diff(y_true, p_a, p_b, n_boot: int = 1000, seed: int = 42) -> dict:
    """Différence de ROC-AUC (a - b) : IC bootstrap à 95 % et p-value approchée bilatérale."""
    y, a, b = np.asarray(y_true), np.asarray(p_a), np.asarray(p_b)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            diffs.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    diffs = np.array(diffs)
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {"diff": float(roc_auc_score(y, a) - roc_auc_score(y, b)),
            "ic_low": float(np.percentile(diffs, 2.5)), "ic_high": float(np.percentile(diffs, 97.5)),
            "p_value": float(min(1.0, p))}


# --------------------------------------------------------------------------- #
# Comparaison des modèles en validation croisée : test de Friedman
# --------------------------------------------------------------------------- #
def fold_scores(searches: dict, metric: str = "ap") -> pd.DataFrame:
    """Score de chaque pli (5) pour la meilleure configuration de chaque modèle."""
    out = {}
    for name, s in searches.items():
        i = s.best_index_
        out[name] = [s.cv_results_[f"split{k}_test_{metric}"][i] for k in range(s.cv.get_n_splits())]
    return pd.DataFrame(out)


def friedman_test(scores: pd.DataFrame) -> dict:
    """Test de Friedman (alternative non paramétrique à l'ANOVA à mesures répétées).

    H0 : les modèles ont le même rang moyen sur les plis. Précaution : seulement
    5 plis, non indépendants (ils partagent des données d'entraînement) -> test
    indicatif, peu puissant.
    """
    stat, p = stats.friedmanchisquare(*[scores[c] for c in scores.columns])
    ranks = scores.rank(axis=1, ascending=False).mean().sort_values()
    return {"statistique": float(stat), "p_value": float(p), "rang_moyen": ranks}
