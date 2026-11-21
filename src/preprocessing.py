"""Prétraitement du dataset Employee Attrition.

Pipeline en deux temps, pour éviter toute fuite de données (data leakage) :

1. ``clean_data``      : nettoyage *sans apprentissage* (types, doublons, plages
                         valides, encodage binaire de la cible). Sans risque de fuite.
2. ``build_preprocessor`` : transformations *avec apprentissage* (imputation,
                         écrêtage des outliers, standardisation, one-hot). Ce
                         transformateur est ajusté (``fit``) UNIQUEMENT sur le
                         jeu d'entraînement, puis appliqué au test.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --------------------------------------------------------------------------- #
# Schéma du dataset
# --------------------------------------------------------------------------- #
ID_COL = "EmployeeID"
TARGET = "Attrition"

# Variables quantitatives continues / discrètes
NUMERIC_COLS = [
    "Age",
    "YearsAtCompany",
    "MonthlyIncome",
    "DistanceFromHome",
    "TrainingHoursLastYear",
]
# Échelles ordinales 1..5 : l'ordre a un sens, on le conserve (pas de one-hot)
ORDINAL_COLS = ["JobLevel", "JobSatisfaction", "WorkLifeBalance", "PerformanceRating"]
# Variables Yes/No -> 0/1
BINARY_COLS = ["OverTime", "PromotionLast5Years"]
# Variable nominale (sans ordre) -> one-hot
NOMINAL_COLS = ["Department"]

FEATURE_COLS = NUMERIC_COLS + ORDINAL_COLS + BINARY_COLS + NOMINAL_COLS

# Plages de valeurs plausibles (une valeur hors plage est traitée comme manquante)
VALID_RANGES: dict[str, tuple[float, float]] = {
    "Age": (18, 70),
    "YearsAtCompany": (0, 50),
    "MonthlyIncome": (0, float("inf")),
    "DistanceFromHome": (0, float("inf")),
    "TrainingHoursLastYear": (0, float("inf")),
    "JobLevel": (1, 5),
    "JobSatisfaction": (1, 5),
    "WorkLifeBalance": (1, 5),
    "PerformanceRating": (1, 5),
}

YES_NO = {"yes": 1, "no": 0}


# --------------------------------------------------------------------------- #
# 1. Chargement et nettoyage (sans apprentissage)
# --------------------------------------------------------------------------- #
def load_raw(path: str | Path) -> pd.DataFrame:
    """Charge le CSV brut."""
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Nettoie le jeu brut et renvoie ``(df_propre, rapport)``.

    Étapes : normalisation des textes, contrôle du schéma, suppression des
    doublons, mise à NaN des valeurs hors plage, encodage Yes/No -> 1/0.
    Aucune statistique du jeu n'est apprise ici.
    """
    df = df.copy()
    report: dict = {"rows_raw": int(len(df)), "columns_raw": int(df.shape[1])}

    df.columns = df.columns.str.strip()
    missing_cols = set(FEATURE_COLS + [TARGET, ID_COL]) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Colonnes manquantes dans le fichier : {sorted(missing_cols)}")

    # Texte : suppression des espaces parasites
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype("string").str.strip()

    # Doublons (lignes identiques, puis identifiants répétés)
    n = len(df)
    df = df.drop_duplicates()
    report["duplicate_rows_removed"] = n - len(df)
    n = len(df)
    df = df.drop_duplicates(subset=ID_COL, keep="first")
    report["duplicate_ids_removed"] = n - len(df)

    # Valeurs hors plage -> NaN (imputées ensuite dans le pipeline, sur le train)
    out_of_range = {}
    for col, (lo, hi) in VALID_RANGES.items():
        df[col] = pd.to_numeric(df[col], errors="coerce")
        bad = (df[col] < lo) | (df[col] > hi)
        if bad.any():
            out_of_range[col] = int(bad.sum())
            df.loc[bad, col] = np.nan
    report["out_of_range_to_nan"] = out_of_range

    # Encodage Yes/No -> 1/0 pour la cible et les variables binaires
    for col in [TARGET] + BINARY_COLS:
        df[col] = df[col].str.lower().map(YES_NO)
    # Une ligne sans cible est inutilisable pour l'apprentissage supervisé
    n = len(df)
    df = df.dropna(subset=[TARGET])
    report["rows_without_target_removed"] = n - len(df)
    df[TARGET] = df[TARGET].astype(int)

    report["missing_after_cleaning"] = {
        c: int(v) for c, v in df[FEATURE_COLS].isna().sum().items() if v > 0
    }
    report["rows_clean"] = int(len(df))
    return df.reset_index(drop=True), report


def iqr_outlier_report(df: pd.DataFrame, cols=NUMERIC_COLS, factors=(1.5, 3.0)) -> pd.DataFrame:
    """Compte, par variable, les valeurs hors des bornes de Tukey (Q1-k*IQR, Q3+k*IQR)."""
    rows = []
    for col in cols:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        row = {"variable": col, "Q1": q1, "Q3": q3}
        for k in factors:
            mask = (df[col] < q1 - k * iqr) | (df[col] > q3 + k * iqr)
            row[f"outliers_k={k}"] = int(mask.sum())
        rows.append(row)
    return pd.DataFrame(rows).set_index("variable")


# --------------------------------------------------------------------------- #
# 2. Transformateurs avec apprentissage (à ajuster sur le train uniquement)
# --------------------------------------------------------------------------- #
class IQRClipper(BaseEstimator, TransformerMixin):
    """Écrête les valeurs hors de [Q1 - k*IQR, Q3 + k*IQR] (winsorisation).

    Les bornes sont apprises dans ``fit`` (donc sur le train seulement). Avec
    ``factor=3.0`` seules les valeurs *extrêmes* sont touchées : les valeurs
    atypiques mais plausibles (ex. 35 ans d'ancienneté) restent quasi intactes.
    """

    def __init__(self, factor: float = 3.0):
        self.factor = factor

    def fit(self, X, y=None):
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        X = np.asarray(X, dtype=float)
        self.n_features_in_ = X.shape[1]
        q1, q3 = np.nanpercentile(X, [25, 75], axis=0)
        iqr = q3 - q1
        self.lower_ = q1 - self.factor * iqr
        self.upper_ = q3 + self.factor * iqr
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = getattr(
                self, "feature_names_in_", [f"x{i}" for i in range(self.n_features_in_)]
            )
        return np.asarray(input_features, dtype=object)


def build_preprocessor(scale: bool = True, clip_outliers: bool = True,
                       iqr_factor: float = 3.0) -> ColumnTransformer:
    """Construit le ColumnTransformer de prétraitement.

    Paramètres
    ----------
    scale : standardisation Z-score (nécessaire pour K-NN, SVM, Naïve Bayes
        gaussien ; inutile mais inoffensive pour les arbres).
    clip_outliers : écrêtage IQR des variables continues.
    iqr_factor : facteur k de l'écrêtage (3.0 = valeurs extrêmes uniquement).
    """
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if clip_outliers:
        num_steps.append(("clip", IQRClipper(factor=iqr_factor)))
    if scale:
        num_steps.append(("scale", StandardScaler()))

    ord_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        ord_steps.append(("scale", StandardScaler()))

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(num_steps), NUMERIC_COLS),
            ("ord", Pipeline(ord_steps), ORDINAL_COLS),
            ("bin", SimpleImputer(strategy="most_frequent"), BINARY_COLS),
            (
                "nom",
                Pipeline([
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]),
                NOMINAL_COLS,
            ),
        ],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")


# --------------------------------------------------------------------------- #
# 3. Découpage train / test
# --------------------------------------------------------------------------- #
def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Découpage stratifié (conserve ~9 % de départs dans train et test).

    Le jeu de test reste intact jusqu'à l'évaluation finale ; le réglage des
    hyperparamètres se fait par validation croisée sur le train uniquement.
    """
    X = df[FEATURE_COLS]
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
