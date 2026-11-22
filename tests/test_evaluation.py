import numpy as np
import pandas as pd

from src.evaluation import (
    best_f1_threshold, bootstrap_ci, fold_scores, friedman_test, metrics_at, paired_bootstrap_diff,
)


def _scores(n=1500, seed=0, signal=1.5):
    rng = np.random.default_rng(seed)
    y = rng.binomial(1, 0.1, n)
    p = 1 / (1 + np.exp(-(signal * y + rng.normal(0, 1, n) - 2.5)))
    return y, p


def test_metrics_at_counts_and_ranges():
    y = np.array([0, 0, 1, 1, 0, 1])
    p = np.array([0.1, 0.6, 0.7, 0.4, 0.2, 0.9])
    m = metrics_at(y, p, 0.5)
    assert (m["TN"], m["FP"], m["FN"], m["TP"]) == (2, 1, 1, 2)
    assert m["precision"] == 2 / 3 and m["recall"] == 2 / 3


def test_best_threshold_beats_default_f1():
    y, p = _scores()
    thr, f1 = best_f1_threshold(y, p)
    assert 0 < thr < 1 and f1 >= metrics_at(y, p, 0.5)["f1"]


def test_bootstrap_ci_contains_point_estimate_and_orders_models():
    y, good = _scores(signal=2.5)
    _, bad = _scores(seed=5, signal=0.0)
    ci = bootstrap_ci(y, {"good": good, "bad": bad}, n_boot=200)
    for _, r in ci.iterrows():
        assert r.roc_auc_low <= r.roc_auc <= r.roc_auc_high
    assert ci.loc["good", "roc_auc"] > ci.loc["bad", "roc_auc"]


def test_paired_bootstrap_detects_better_model():
    y, good = _scores(signal=3.0)
    rng = np.random.default_rng(9)
    r = paired_bootstrap_diff(y, good, rng.random(len(y)), n_boot=200)
    assert r["diff"] > 0.2 and r["p_value"] < 0.05


def test_friedman_detects_consistent_ranking():
    scores = pd.DataFrame({"A": [0.9, 0.91, 0.92, 0.9, 0.93], "B": [0.5, 0.52, 0.51, 0.5, 0.53],
                           "C": [0.1, 0.12, 0.11, 0.1, 0.13]})
    assert friedman_test(scores)["p_value"] < 0.05
