"""Tests statistiques pour l'analyse exploratoire (EDA).

Chaque fonction renvoie un DataFrame prêt à être affiché ou sauvegardé.
La correction des p-values multiples (Benjamini-Hochberg, FDR) s'applique
avec ``add_fdr`` sur une *famille* de tests menés sur la même question.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


# --------------------------------------------------------------------------- #
# Correction des comparaisons multiples
# --------------------------------------------------------------------------- #
def add_fdr(df: pd.DataFrame, p_col: str = "p_value", alpha: float = 0.05) -> pd.DataFrame:
    """Ajoute la p-value ajustée Benjamini-Hochberg et la décision à alpha."""
    out = df.copy()
    out["p_adj_BH"] = stats.false_discovery_control(out[p_col].to_numpy(), method="bh")
    out["significatif"] = out["p_adj_BH"] < alpha
    return out


# --------------------------------------------------------------------------- #
# Normalité
# --------------------------------------------------------------------------- #
def shapiro_table(df: pd.DataFrame, cols, sample_size: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Shapiro-Wilk + asymétrie / aplatissement.

    Shapiro-Wilk est fiable jusqu'à ~5000 observations (avertissement scipy
    au-delà) : on teste donc un échantillon aléatoire reproductible.
    """
    rows = []
    for col in cols:
        x = df[col].dropna()
        xs = x.sample(min(sample_size, len(x)), random_state=seed)
        w, p = stats.shapiro(xs)
        rows.append({
            "variable": col, "n_teste": len(xs), "W": w, "p_value": p,
            "asymetrie": stats.skew(x), "aplatissement": stats.kurtosis(x),
            "normale (p>0.05)": bool(p > 0.05),
        })
    return pd.DataFrame(rows).set_index("variable")


# --------------------------------------------------------------------------- #
# Variable quantitative / ordinale  vs  cible binaire : Mann-Whitney U
# --------------------------------------------------------------------------- #
def mannwhitney_table(df: pd.DataFrame, cols, target: str) -> pd.DataFrame:
    """Mann-Whitney U bilatéral entre départs (1) et maintiens (0).

    Taille d'effet : corrélation bisériale par rangs r = 2*U1/(n1*n0) - 1,
    comprise entre -1 et 1 ; r > 0 : les valeurs sont plutôt plus élevées
    chez ceux qui partent. Repères usuels : |r| ~0.1 faible, 0.3 moyen, 0.5 fort.
    """
    g1, g0 = df[df[target] == 1], df[df[target] == 0]
    rows = []
    for col in cols:
        a, b = g1[col].dropna(), g0[col].dropna()
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        rows.append({
            "variable": col,
            "mediane_part": a.median(), "mediane_reste": b.median(),
            "U": u, "p_value": p,
            "effet_r_biserial": 2 * u / (len(a) * len(b)) - 1,
        })
    return pd.DataFrame(rows).set_index("variable")


# --------------------------------------------------------------------------- #
# Variable catégorielle  vs  cible binaire : khi-deux + V de Cramér
# --------------------------------------------------------------------------- #
def chi2_table(df: pd.DataFrame, cols, target: str) -> pd.DataFrame:
    """Test d'indépendance du khi-deux et V de Cramér (taille d'effet).

    Vérifie la condition de validité : effectifs théoriques >= 5 partout.
    """
    rows = []
    for col in cols:
        ct = pd.crosstab(df[col], df[target])
        chi2, p, dof, expected = stats.chi2_contingency(ct)
        n = ct.to_numpy().sum()
        v = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))
        rows.append({
            "variable": col, "chi2": chi2, "ddl": dof, "p_value": p,
            "V_Cramer": v, "min_effectif_theorique": float(expected.min()),
            "condition_valide (>=5)": bool(expected.min() >= 5),
        })
    return pd.DataFrame(rows).set_index("variable")


# --------------------------------------------------------------------------- #
# Plus de 2 groupes : Kruskal-Wallis
# --------------------------------------------------------------------------- #
def kruskal_table(df: pd.DataFrame, value_col: str, group_cols) -> pd.DataFrame:
    """Kruskal-Wallis (alternative non paramétrique à l'ANOVA) + epsilon².

    epsilon² = (H - k + 1) / (n - k) : part de variance expliquée par rangs.
    """
    rows = []
    for g in group_cols:
        groups = [s[value_col].dropna().to_numpy() for _, s in df.groupby(g)]
        h, p = stats.kruskal(*groups)
        n, k = sum(len(x) for x in groups), len(groups)
        rows.append({
            "variable_quanti": value_col, "facteur": g, "k_groupes": k,
            "H": h, "p_value": p, "epsilon2": (h - k + 1) / (n - k),
        })
    return pd.DataFrame(rows).set_index("facteur")


# --------------------------------------------------------------------------- #
# Corrélations de Spearman
# --------------------------------------------------------------------------- #
def spearman_pairs(df: pd.DataFrame, cols) -> pd.DataFrame:
    """Toutes les paires de colonnes : rho de Spearman et p-value."""
    cols = list(cols)
    rows = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            rho, p = stats.spearmanr(df[a], df[b])
            rows.append({"var_1": a, "var_2": b, "rho": rho, "p_value": p})
    out = pd.DataFrame(rows)
    out["|rho|"] = out["rho"].abs()
    return out.sort_values("|rho|", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Interprétation de la taille d'effet
# --------------------------------------------------------------------------- #
def effect_label(x: float) -> str:
    """Étiquette qualitative (repères de Cohen) pour |r| ou V de Cramér."""
    x = abs(x)
    if x < 0.1:
        return "négligeable"
    if x < 0.3:
        return "faible"
    if x < 0.5:
        return "moyen"
    return "fort"
