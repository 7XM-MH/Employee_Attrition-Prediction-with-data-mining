import numpy as np
import pandas as pd

from src.stats_tests import (
    add_fdr, chi2_table, effect_label, kruskal_table,
    mannwhitney_table, shapiro_table, spearman_pairs,
)


def _df(n=2000, seed=1):
    rng = np.random.default_rng(seed)
    y = rng.binomial(1, 0.3, n)
    return pd.DataFrame({
        "y": y,
        "gauss": rng.normal(0, 1, n),
        "expo": rng.exponential(1, n),
        "shift": rng.normal(0, 1, n) + 0.8 * y,          # dépend de y
        "cat_dep": np.where(rng.random(n) < 0.2 + 0.4 * y, "A", "B"),
        "cat_ind": rng.choice(["A", "B"], n),             # indépendante de y
        "grp": rng.choice([1, 2, 3], n),
    })


def test_shapiro_detects_non_normality():
    t = shapiro_table(_df(), ["gauss", "expo"])
    assert bool(t.loc["gauss", "normale (p>0.05)"]) and not bool(t.loc["expo", "normale (p>0.05)"])


def test_mannwhitney_detects_shift_and_sign():
    t = mannwhitney_table(_df(), ["shift", "gauss"], "y")
    assert t.loc["shift", "p_value"] < 1e-10 and t.loc["shift", "effet_r_biserial"] > 0.3
    assert t.loc["gauss", "p_value"] > 0.001


def test_chi2_dependent_vs_independent():
    t = chi2_table(_df(), ["cat_dep", "cat_ind"], "y")
    assert t.loc["cat_dep", "p_value"] < 1e-10 and t.loc["cat_ind", "p_value"] > 0.001
    assert 0 <= t["V_Cramer"].min() and t["V_Cramer"].max() <= 1


def test_fdr_adjusted_pvalues_not_smaller_than_raw():
    p = pd.DataFrame({"p_value": [0.001, 0.02, 0.04, 0.5]})
    out = add_fdr(p)
    assert (out["p_adj_BH"] >= out["p_value"]).all() and out["p_adj_BH"].max() <= 1


def test_kruskal_and_spearman():
    k = kruskal_table(_df(), "shift", ["grp"])
    assert k.loc["grp", "k_groupes"] == 3
    s = spearman_pairs(_df(), ["gauss", "shift", "y"])
    assert s.iloc[0]["|rho|"] >= s.iloc[-1]["|rho|"]


def test_effect_label():
    assert [effect_label(v) for v in (0.05, 0.2, 0.4, 0.7)] == ["négligeable", "faible", "moyen", "fort"]
