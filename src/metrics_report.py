"""Métriques comparatives détaillées (critère « analyse des métriques comparatives »).

Toutes les fonctions travaillent sur des **probabilités calibrées** de la classe
« part » et sur un seuil de décision fixé *avant* d'examiner le test (choisi sur
le train). Convention : classe positive = départ (1).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.preprocessing import OrdinalEncoder

from src.preprocessing import BINARY_COLS, NOMINAL_COLS, NUMERIC_COLS, ORDINAL_COLS

# Métriques dépendantes du seuil / des scores ; sens d'interprétation
THRESHOLD_METRICS = ["accuracy", "balanced_accuracy", "precision", "recall", "specificity",
                     "f1", "f1_macro", "mcc", "kappa"]
SCORE_METRICS = ["roc_auc", "pr_auc", "brier", "log_loss"]
ALL_METRICS = THRESHOLD_METRICS + SCORE_METRICS
LOWER_IS_BETTER = {"brier", "log_loss"}


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def threshold_metrics(y, p, threshold: float) -> dict:
    """Métriques issues de la matrice de confusion au seuil donné (calcul numpy, rapide)."""
    y = np.asarray(y).astype(int)
    pred = (np.asarray(p) >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    n = tp + tn + fp + fn
    precision, recall, spec = _safe_div(tp, tp + fp), _safe_div(tp, tp + fn), _safe_div(tn, tn + fp)
    f1_pos = _safe_div(2 * tp, 2 * tp + fp + fn)
    f1_neg = _safe_div(2 * tn, 2 * tn + fp + fn)
    denom = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    pe = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / n ** 2
    acc = (tp + tn) / n
    return {
        "accuracy": acc, "balanced_accuracy": (recall + spec) / 2, "precision": precision,
        "recall": recall, "specificity": spec, "f1": f1_pos, "f1_macro": (f1_pos + f1_neg) / 2,
        "mcc": (tp * tn - fp * fn) / denom if denom else 0.0,
        "kappa": (acc - pe) / (1 - pe) if pe < 1 else 0.0,
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
    }


def score_metrics(y, p) -> dict:
    """Métriques indépendantes du seuil : discrimination (AUC) et qualité des probabilités."""
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    pc = np.clip(p, 1e-15, 1 - 1e-15)
    return {
        "roc_auc": roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan,
        "pr_auc": average_precision_score(y, p) if y.sum() else np.nan,
        "brier": float(np.mean((p - y) ** 2)),
        "log_loss": float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))),
    }


def full_metrics(y, p, threshold: float) -> dict:
    return {**threshold_metrics(y, p, threshold), **score_metrics(y, p)}


# --------------------------------------------------------------------------- #
# Incertitude : bootstrap apparié (mêmes rééchantillons pour tous les modèles)
# --------------------------------------------------------------------------- #
def make_bootstrap_indices(y, n_boot: int = 1000, seed: int = 42) -> list[np.ndarray]:
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    idx = [rng.integers(0, len(y), len(y)) for _ in range(n_boot)]
    return [i for i in idx if 0 < y[i].sum() < len(i)]


def bootstrap_metrics(y, p, threshold: float, indices: list[np.ndarray], alpha: float = 0.05) -> pd.DataFrame:
    """Estimation ponctuelle et IC à 95 % (percentiles) de toutes les métriques."""
    y, p = np.asarray(y), np.asarray(p)
    est = full_metrics(y, p, threshold)
    draws = pd.DataFrame([full_metrics(y[i], p[i], threshold) for i in indices])[ALL_METRICS]
    lo, hi = draws.quantile(alpha / 2), draws.quantile(1 - alpha / 2)
    return pd.DataFrame({"estimation": pd.Series({m: est[m] for m in ALL_METRICS}), "ic_bas": lo, "ic_haut": hi})


# --------------------------------------------------------------------------- #
# Performance à budget d'alertes fixé
# --------------------------------------------------------------------------- #
def alert_budget_metrics(y, p, fractions=(0.05, 0.10, 0.20, 0.30)) -> pd.DataFrame:
    """Précision, rappel et lift quand on alerte les ``k`` % d'employés les plus à risque.

    Le classement est indépendant du seuil ; les égalités de score sont départagées
    par l'ordre des données (stable).
    """
    y, p = np.asarray(y).astype(int), np.asarray(p)
    order = np.argsort(-p, kind="mergesort")
    prev, rows = y.mean(), []
    for f in fractions:
        n_top = int(np.ceil(f * len(y)))
        tp = int(y[order[:n_top]].sum())
        rows.append({"k": f, "alertes": n_top, "départs_captés": tp, "précision": tp / n_top,
                     "rappel": tp / y.sum(), "lift": (tp / n_top) / prev})
    return pd.DataFrame(rows).set_index("k")


def gain_curve(y, p) -> tuple[np.ndarray, np.ndarray]:
    """Courbe de gain cumulé : (% d'employés contactés, % de départs captés)."""
    y, p = np.asarray(y).astype(int), np.asarray(p)
    order = np.argsort(-p, kind="mergesort")
    cum = np.concatenate([[0], np.cumsum(y[order])]) / y.sum()
    return np.arange(len(y) + 1) / len(y), cum


# --------------------------------------------------------------------------- #
# Classement multicritère
# --------------------------------------------------------------------------- #
def rank_table(metrics: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Rang de chaque modèle (1 = meilleur) sur chaque métrique ; Brier / log-loss : plus bas = mieux."""
    ranks = pd.DataFrame(index=metrics.index)
    for c in columns:
        ranks[c] = metrics[c].rank(ascending=c in LOWER_IS_BETTER, method="min")
    ranks["rang_moyen"] = ranks[columns].mean(axis=1)
    return ranks.sort_values("rang_moyen")


# --------------------------------------------------------------------------- #
# Bandes de confiance des courbes ROC et précision-rappel
# --------------------------------------------------------------------------- #
def roc_band(y, p, indices, grid=None, alpha: float = 0.05):
    """Courbe ROC et bande de confiance bootstrap (TPR interpolé sur une grille de FPR)."""
    y, p = np.asarray(y), np.asarray(p)
    grid = np.linspace(0, 1, 101) if grid is None else grid
    fpr, tpr, _ = roc_curve(y, p)
    tprs = []
    for i in indices:
        f, t, _ = roc_curve(y[i], p[i])
        tprs.append(np.interp(grid, f, t))
    tprs = np.array(tprs)
    return fpr, tpr, grid, np.quantile(tprs, alpha / 2, axis=0), np.quantile(tprs, 1 - alpha / 2, axis=0)


def pr_band(y, p, indices, grid=None, alpha: float = 0.05):
    """Courbe précision-rappel et bande de confiance bootstrap (précision interpolée sur une grille de rappel)."""
    y, p = np.asarray(y), np.asarray(p)
    grid = np.linspace(0, 1, 101) if grid is None else grid
    prec, rec, _ = precision_recall_curve(y, p)
    precs = []
    for i in indices:
        pr_, rc_, _ = precision_recall_curve(y[i], p[i])
        precs.append(np.interp(grid, rc_[::-1], pr_[::-1]))
    precs = np.array(precs)
    return prec, rec, grid, np.quantile(precs, alpha / 2, axis=0), np.quantile(precs, 1 - alpha / 2, axis=0)


# --------------------------------------------------------------------------- #
# Importance des variables : plusieurs perspectives, au niveau des 12 variables d'origine
# --------------------------------------------------------------------------- #
def mutual_information_raw(X: pd.DataFrame, y, seed: int = 0) -> pd.Series:
    """Information mutuelle de chaque variable d'origine avec la cible (variables discrètes déclarées comme telles)."""
    Xe = X.copy()
    Xe[NOMINAL_COLS] = OrdinalEncoder().fit_transform(Xe[NOMINAL_COLS])
    discrete = [c not in NUMERIC_COLS for c in Xe.columns]
    mi = mutual_info_classif(Xe, y, discrete_features=discrete, random_state=seed)
    return pd.Series(mi, index=Xe.columns)


def aggregate_importance(names, values) -> pd.Series:
    """Ramène les colonnes encodées (``Department_*``) à la variable d'origine en sommant leurs importances."""
    s = pd.Series(np.asarray(values, dtype=float), index=[str(n) for n in names])
    keys = [("Department" if n.startswith("Department_") else n) for n in s.index]
    return s.groupby(keys).sum()


def normalize_01(s: pd.Series) -> pd.Series:
    """Ramène à [0, 1] (valeurs négatives = bruit, ramenées à 0) pour comparer des méthodes différentes."""
    s = s.clip(lower=0)
    return s / s.max() if s.max() > 0 else s
