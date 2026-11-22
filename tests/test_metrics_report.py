import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef

from src.metrics_report import (
    LOWER_IS_BETTER, aggregate_importance, alert_budget_metrics, bootstrap_metrics, full_metrics,
    gain_curve, make_bootstrap_indices, normalize_01, rank_table, roc_band, threshold_metrics,
)


def _data(n=2000, seed=0, signal=1.5):
    rng = np.random.default_rng(seed)
    y = rng.binomial(1, 0.1, n)
    p = 1 / (1 + np.exp(-(signal * y + rng.normal(0, 1, n) - 2.5)))
    return y, p


def test_threshold_metrics_match_sklearn_and_hand_counts():
    y, p = _data()
    m = threshold_metrics(y, p, 0.2)
    pred = (p >= 0.2).astype(int)
    assert m["mcc"] == pytest.approx(matthews_corrcoef(y, pred))
    assert m["kappa"] == pytest.approx(cohen_kappa_score(y, pred))
    assert m["TP"] + m["TN"] + m["FP"] + m["FN"] == len(y)
    assert m["balanced_accuracy"] == pytest.approx((m["recall"] + m["specificity"]) / 2)


def test_degenerate_classifier_has_zero_recall_and_no_division_error():
    y, _ = _data()
    m = threshold_metrics(y, np.full(len(y), 0.1), 0.5)       # n'alerte jamais
    assert m["recall"] == 0 and m["precision"] == 0 and m["f1"] == 0 and m["mcc"] == 0


def test_bootstrap_ci_brackets_estimate():
    y, p = _data()
    idx = make_bootstrap_indices(y, n_boot=150)
    ci = bootstrap_metrics(y, p, 0.15, idx)
    inside = (ci["ic_bas"] <= ci["estimation"] + 1e-9) & (ci["estimation"] <= ci["ic_haut"] + 1e-9)
    assert inside.mean() > 0.9 and (ci["ic_bas"] <= ci["ic_haut"]).all()


def test_alert_budget_perfect_scores_and_gain_curve():
    y = np.array([1] * 10 + [0] * 90)
    p = np.concatenate([np.linspace(0.9, 0.8, 10), np.linspace(0.1, 0.0, 90)])
    t = alert_budget_metrics(y, p, fractions=(0.1, 0.2))
    assert t.loc[0.1, "précision"] == 1 and t.loc[0.1, "rappel"] == 1 and t.loc[0.1, "lift"] == pytest.approx(10)
    x, g = gain_curve(y, p)
    assert x[0] == 0 and g[0] == 0 and g[-1] == 1 and np.all(np.diff(g) >= 0)


def test_rank_table_direction_for_lower_is_better_metrics():
    m = pd.DataFrame({"roc_auc": [0.7, 0.6], "brier": [0.08, 0.07]}, index=["A", "B"])
    r = rank_table(m, ["roc_auc", "brier"])
    assert r.loc["A", "roc_auc"] == 1 and r.loc["B", "brier"] == 1 and "brier" in LOWER_IS_BETTER


def test_aggregate_and_normalize_importance():
    s = aggregate_importance(["Age", "Department_IT", "Department_HR"], [0.2, 0.1, 0.3])
    assert s["Department"] == pytest.approx(0.4) and s["Age"] == 0.2
    n = normalize_01(pd.Series({"a": 2.0, "b": -1.0, "c": 1.0}))
    assert n.max() == 1 and n["b"] == 0


def test_roc_band_contains_curve_roughly():
    y, p = _data()
    idx = make_bootstrap_indices(y, n_boot=60)
    fpr, tpr, grid, lo, hi = roc_band(y, p, idx)
    assert (lo <= hi).all() and lo[0] <= 0.2 and hi[-1] >= 0.95
