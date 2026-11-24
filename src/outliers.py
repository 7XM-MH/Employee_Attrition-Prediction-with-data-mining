"""Détection des outliers : Z-score, IQR, Isolation Forest et Local Outlier Factor.

Ce module est **descriptif** : les quatre méthodes sont appliquées au jeu brut
afin de produire un rapport pédagogique. Aucune cible (``Attrition``) ni aucun
identifiant (``EmployeeID``) n'est utilisé pour détecter les outliers.

Règle de non-fuite : si une méthode devait un jour alimenter un modèle, ses
statistiques (moyenne/écart-type, quantiles, standardisation, ajustement des
modèles) devraient être apprises sur le jeu d'entraînement uniquement, puis
appliquées au test. Ici le rapport est calculé sur le jeu complet et doit être
lu comme une analyse exploratoire, jamais comme une étape du pipeline ML.

L'IQR n'est pas réimplémentée : :func:`iqr_outliers` délègue à
:func:`src.preprocessing.iqr_outlier_report`. La winsorisation utilisée par le
pipeline reste ``IQRClipper`` dans ``src/preprocessing.py``.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from src.preprocessing import NUMERIC_COLS, ORDINAL_COLS, iqr_outlier_report

#: Schéma unique de tous les rapports produits par ce module.
REPORT_COLUMNS: list[str] = [
    "variable",
    "n_outliers",
    "pct_outliers",
    "seuil",
    "action_recommandee",
]

#: Schéma consolidé renvoyé par :func:`outlier_summary`.
SUMMARY_COLUMNS: list[str] = [
    "methode",
    "variable",
    "n_outliers",
    "pct_outliers",
    "seuil",
    "action_recommandee",
]

_UNIVARIATE_ACTION = "winsorisation après vérification manuelle"
_MULTIVARIATE_ACTION = "investigation ; suppression uniquement si erreur avérée"


def _as_numeric_series(values: pd.Series) -> pd.Series:
    """Convertit une colonne en numérique, les valeurs invalides devenant NaN."""
    return pd.to_numeric(values, errors="coerce")


def _counts_frame(rows: list[dict], columns: Sequence[str] = REPORT_COLUMNS) -> pd.DataFrame:
    """Construit un DataFrame au schéma stable, même si ``rows`` est vide."""
    return pd.DataFrame(rows, columns=list(columns))


def _count_row(variable: str, mask: pd.Series, n_rows: int, seuil: str,
               action: str) -> dict:
    """Crée une ligne de rapport à partir d'un masque booléen."""
    return _count_from_n(variable, int(mask.fillna(False).sum()), n_rows, seuil, action)


def _count_from_n(variable: str, n_outliers: int, n_rows: int, seuil: str,
                  action: str) -> dict:
    """Crée une ligne de rapport à partir d'un effectif déjà calculé."""
    pct = round(100.0 * n_outliers / n_rows, 2) if n_rows else 0.0
    return {
        "variable": variable,
        "n_outliers": n_outliers,
        "pct_outliers": pct,
        "seuil": seuil,
        "action_recommandee": action,
    }


def zscore_outliers(df: pd.DataFrame, cols: Sequence[str],
                    threshold: float = 3.0) -> pd.DataFrame:
    """Compte les valeurs dont le Z-score absolu dépasse ``threshold``.

    Paramètres
    ----------
    df : pandas.DataFrame
        Jeu observé (brut pour le rapport descriptif).
    cols : sequence of str
        Variables numériques à examiner. ``EmployeeID`` et la cible doivent
        rester hors de cette liste.
    threshold : float
        Seuil sur ``|z|`` ; 3.0 corresponds à la règle du cours.

    Renvoie
    -------
    pandas.DataFrame
        Colonnes : ``variable``, ``n_outliers``, ``pct_outliers``, ``seuil``,
        ``action_recommandee``.
    """
    if df.empty:
        raise ValueError("Le DataFrame fourni est vide.")
    rows = []
    seuil = f"|z| > {threshold:g}"
    for col in cols:
        s = _as_numeric_series(df[col])
        mean = s.mean()
        std = s.std(ddof=0)
        if not np.isfinite(std) or std == 0:
            mask = pd.Series(False, index=s.index)
        else:
            mask = ((s - mean).abs() / std) > threshold
        rows.append(_count_row(col, mask, len(s), seuil, _UNIVARIATE_ACTION))
    return _counts_frame(rows)


def iqr_outliers(df: pd.DataFrame, cols: Sequence[str] = NUMERIC_COLS,
                 factor: float = 1.5) -> pd.DataFrame:
    """Compte les valeurs hors des bornes de Tukey ``[Q1-k*IQR, Q3+k*IQR]``.

    Cette fonction **délègue** le calcul à
    :func:`src.preprocessing.iqr_outlier_report` : la règle IQR n'est pas
    dupliquée ici. Le facteur 1.5 est la règle de Tukey ; le facteur 3.0 reste
    utilisé par ``IQRClipper`` pour l'écrêtage extrême du pipeline ML.
    """
    base = iqr_outlier_report(df, cols=cols, factors=(factor,))
    key = f"outliers_k={factor}"
    rows = [
        _count_from_n(
            str(variable),
            int(base.loc[variable, key]),
            len(df),
            f"IQR x{factor:g}",
            _UNIVARIATE_ACTION,
        )
        for variable in base.index
    ]
    return _counts_frame(rows)


def _prepare_matrix(X) -> tuple[np.ndarray, pd.Index]:
    """Convertit ``X`` en matrice numérique imputée et standardisée."""
    frame = pd.DataFrame(X).copy()
    if frame.empty or frame.shape[1] == 0:
        raise ValueError("La matrice de variables est vide.")
    frame = frame.apply(_as_numeric_series)
    matrix = SimpleImputer(strategy="median").fit_transform(frame)
    matrix = StandardScaler().fit_transform(matrix)
    return np.asarray(matrix, dtype=float), frame.index


def isolation_forest_outliers(X, contamination: float = 0.05,
                              random_state: int = 42) -> pd.Series:
    """Détecte les anomalies multivariées avec ``IsolationForest``.

    ``X`` ne doit contenir que des variables explicatives numériques. La
    standardisation et l'imputation sont ajustées sur les données fournies :
    pour un usage modèle, elles doivent être ajustées sur le train seulement.

    Renvoie une série booléenne indexée comme ``X`` (``True`` = outlier).
    """
    matrix, index = _prepare_matrix(X)
    model = IsolationForest(
        contamination=contamination, random_state=random_state, n_estimators=100
    )
    return pd.Series(model.fit_predict(matrix) == -1, index=index, name="is_outlier")


def lof_outliers(X, n_neighbors: int = 20,
                 contamination: float = 0.05) -> pd.Series:
    """Détecte les anomalies locales avec ``LocalOutlierFactor``.

    ``n_neighbors`` est automatiquement ramené à ``n_samples - 1`` pour les
    petits jeux synthétiques ; sur le jeu réel (10 000 lignes) la valeur par
    défaut de 20 est conservée. LOF est déterministe (pas de graine aléatoire).

    Renvoie une série booléenne indexée comme ``X`` (``True`` = outlier).
    """
    matrix, index = _prepare_matrix(X)
    k = min(int(n_neighbors), max(1, len(index) - 1))
    model = LocalOutlierFactor(n_neighbors=k, contamination=contamination)
    return pd.Series(model.fit_predict(matrix) == -1, index=index, name="is_outlier")


def _multivariate_row(methode: str, mask: pd.Series, n_rows: int, seuil: str,
                      n_vars: int) -> dict:
    """Construit la ligne consolidée d'une méthode multivariée."""
    row = _count_row(
        f"multivarié ({n_vars} variables)", mask, n_rows, seuil, _MULTIVARIATE_ACTION
    )
    return {"methode": methode, **row}


def outlier_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Consolide les quatre méthodes en un seul rapport homogène.

    Les variables examinées sont ``NUMERIC_COLS + ORDINAL_COLS`` effectivement
    présentes dans ``df``. ``EmployeeID`` et ``Attrition`` sont exclus par
    construction : ils ne servent ni au calcul ni à l'interprétation.

    Renvoie un DataFrame avec les colonnes ``methode``, ``variable``,
    ``n_outliers``, ``pct_outliers``, ``seuil`` et ``action_recommandee``.
    """
    cols = [c for c in (list(NUMERIC_COLS) + list(ORDINAL_COLS)) if c in df.columns]
    if not cols:
        raise ValueError("Aucune variable numérique/ordinale attendue n'est présente.")
    n = len(df)
    if n == 0:
        raise ValueError("Le DataFrame fourni est vide.")

    zscore = zscore_outliers(df, cols)
    for frame in (zscore,):
        frame.insert(0, "methode", "Z-score")

    iqr = iqr_outliers(df, cols)
    iqr.insert(0, "methode", "IQR")

    X = df.loc[:, cols]
    k = min(20, max(1, n - 1))
    multivariate = [
        _multivariate_row(
            "Isolation Forest",
            isolation_forest_outliers(X, contamination=0.05, random_state=42),
            n,
            "contamination=0.05, random_state=42",
            len(cols),
        ),
        _multivariate_row(
            "LOF",
            lof_outliers(X, n_neighbors=k, contamination=0.05),
            n,
            f"n_neighbors={k}, contamination=0.05",
            len(cols),
        ),
    ]

    summary = pd.concat(
        [zscore, iqr, pd.DataFrame(multivariate, columns=SUMMARY_COLUMNS)],
        ignore_index=True,
    )
    return summary.loc[:, SUMMARY_COLUMNS]
