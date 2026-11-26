"""Modèles de classification, grilles d'hyperparamètres et fonctions de prédiction.

Parcours A (classification supervisée) du cahier des charges :
Zero-R, K-NN, Arbre de décision, Random Forest, Naïve Bayes, SVM
+ extensions autorisées (modèles ML classiques) : régression logistique,
Gradient Boosting par histogrammes. Aucun Deep Learning, aucun LLM.

Tous les modèles sont des ``Pipeline`` (prétraitement + classifieur) ajustés
sur les données *non transformées* (``X_train_raw.csv``) : à chaque pli de la
validation croisée, imputation / écrêtage / standardisation / encodage sont
réappris sur les seuls plis d'entraînement -> aucune fuite de données.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

from src.preprocessing import (
    BINARY_COLS, FEATURE_COLS, NOMINAL_COLS, NUMERIC_COLS, ORDINAL_COLS,
    build_preprocessor,
)

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
CACHE_DIR = MODELS_DIR / "cache"
RESULTS_DIR = ROOT / "results"
RANDOM_STATE = 42

# Validation croisée K-Fold stratifiée, K = 5 (minimum exigé : K >= 5)
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# Cible déséquilibrée (~9 % de départs) : on optimise la précision moyenne
# (average precision = aire sous la courbe précision-rappel), indépendante du
# seuil. ROC-AUC et F1 sont suivis en parallèle.
SCORING = {"ap": "average_precision", "roc_auc": "roc_auc", "f1": "f1"}
REFIT = "ap"

BASELINE = "Zero-R"
EXTENSIONS = [
    "Régression logistique",
    "Gradient Boosting",
    "Extra Trees",
    "AdaBoost",
    "Bagging",
    "K-NN (cosinus)",
    "Sélection de variables",
]
if XGBClassifier is not None:
    EXTENSIONS.append("XGBoost")


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


# --------------------------------------------------------------------------- #
# Prétraitements spécifiques
# --------------------------------------------------------------------------- #
def build_binned_preprocessor() -> ColumnTransformer:
    """Discrétisation + one-hot : toutes les variables deviennent binaires.

    Utilisé par BernoulliNB (modèle d'événements binaires). Le lissage de
    Laplace (alpha = 1) évite les probabilités nulles pour les modalités rares.
    """
    return ColumnTransformer(
        [
            ("num", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("kbins", KBinsDiscretizer(n_bins=5, encode="onehot-dense", strategy="quantile")),
            ]), NUMERIC_COLS),
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), ORDINAL_COLS + NOMINAL_COLS),
            ("bin", SimpleImputer(strategy="most_frequent"), BINARY_COLS),
        ]
    )


def _tree_preprocessor() -> ColumnTransformer:
    """Arbres : insensibles à l'échelle et aux outliers -> pas de scaling/écrêtage."""
    return build_preprocessor(scale=False, clip_outliers=False)


def get_zero_r() -> Pipeline:
    """Classifieur de référence Zero-R : prédit toujours la classe majoritaire."""
    return Pipeline([("prep", _tree_preprocessor()), ("clf", DummyClassifier(strategy="prior"))])


def get_knn_pipeline(scale: bool = True) -> Pipeline:
    """K-NN avec ou sans standardisation (pour l'étude de l'effet du scaling)."""
    return Pipeline([
        ("prep", build_preprocessor(scale=scale, clip_outliers=scale)),
        ("clf", KNeighborsClassifier()),
    ])


# --------------------------------------------------------------------------- #
# Recherches d'hyperparamètres (GridSearchCV / RandomizedSearchCV, K = 5)
# --------------------------------------------------------------------------- #
def get_searches(n_jobs: int = 1) -> dict:
    """Renvoie un dictionnaire ``nom -> recherche`` (non ajustée)."""
    kw = dict(scoring=SCORING, refit=REFIT, cv=CV, n_jobs=n_jobs)

    searches = {}

    # --- 1. K-NN : étude de K, de la métrique de distance et de la pondération
    searches["K-NN"] = GridSearchCV(
        get_knn_pipeline(scale=True),
        {
            "clf__n_neighbors": [11, 31, 51, 101, 151, 201, 301, 500, 800],
            "clf__metric": ["euclidean", "manhattan"],
            "clf__weights": ["uniform", "distance"],
        },
        **kw,
    )

    # --- 2. Arbre de décision : indice de Gini / gain d'information, élagage
    searches["Arbre de décision"] = GridSearchCV(
        Pipeline([
            ("prep", _tree_preprocessor()),
            ("clf", DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
        {
            "clf__criterion": ["gini", "entropy"],          # Gini vs gain d'information
            "clf__max_depth": [2, 3, 4, 5, 7, 10],          # pré-élagage
            "clf__min_samples_leaf": [10, 20, 50, 100],
            "clf__ccp_alpha": [0.0, 0.0005, 0.002],         # post-élagage (coût-complexité)
        },
        **kw,
    )

    # --- 3. Random Forest (recherche aléatoire : coût de calcul réduit)
    searches["Random Forest"] = RandomizedSearchCV(
        Pipeline([
            ("prep", _tree_preprocessor()),
            ("clf", RandomForestClassifier(class_weight="balanced_subsample", random_state=RANDOM_STATE)),
        ]),
        {
            "clf__n_estimators": [200, 300],
            "clf__max_depth": [2, 3, 4, 6, 8],
            "clf__min_samples_leaf": [10, 20, 40, 80],
            "clf__max_features": ["sqrt", 0.5],
            "clf__criterion": ["gini", "entropy"],
        },
        n_iter=15, random_state=RANDOM_STATE, **kw,
    )

    # --- 4a. Naïve Bayes gaussien (lissage de variance var_smoothing)
    searches["Naive Bayes (Gaussian)"] = GridSearchCV(
        Pipeline([("prep", build_preprocessor(scale=True)), ("clf", GaussianNB())]),
        {"clf__var_smoothing": np.logspace(-9, 1, 11)},
        **kw,
    )

    # --- 4b. Naïve Bayes de Bernoulli avec lissage de Laplace (alpha = 1)
    searches["Naive Bayes (Bernoulli)"] = GridSearchCV(
        Pipeline([("prep", build_binned_preprocessor()), ("clf", BernoulliNB(binarize=None))]),
        {
            "prep__num__kbins__n_bins": [2, 3, 5],
            "clf__alpha": [0.5, 1.0, 5.0, 10.0, 20.0, 50.0],   # alpha = 1.0 : lissage de Laplace
        },
        **kw,
    )

    # --- 5. SVM : noyaux linéaire / RBF / polynomial, réglage de C et gamma
    searches["SVM"] = GridSearchCV(
        Pipeline([
            ("prep", build_preprocessor(scale=True)),
            ("clf", SVC(class_weight="balanced", cache_size=500, random_state=RANDOM_STATE)),
        ]),
        [
            {"clf__kernel": ["linear"], "clf__C": [0.01, 0.1, 1]},
            {"clf__kernel": ["rbf"], "clf__C": [0.1, 1, 10], "clf__gamma": [0.0001, 0.0005, 0.001, 0.005, 0.01]},
            {"clf__kernel": ["poly"], "clf__C": [0.1, 1], "clf__degree": [2, 3], "clf__gamma": ["scale"]},
        ],
        **kw,
    )

    # --- Extensions (ML classique)
    searches["Régression logistique"] = GridSearchCV(
        Pipeline([
            ("prep", build_preprocessor(scale=True)),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE)),
        ]),
        {"clf__C": [0.001, 0.01, 0.1, 1, 10]},
        **kw,
    )
    searches["Gradient Boosting"] = GridSearchCV(
        Pipeline([
            ("prep", _tree_preprocessor()),
            ("clf", HistGradientBoostingClassifier(class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
        {
            "clf__learning_rate": [0.01, 0.03, 0.1],
            "clf__max_depth": [1, 2, 3],
            "clf__max_iter": [50, 100, 200],
            "clf__min_samples_leaf": [20, 50, 100],
        },
        **kw,
    )
    # --- 6. K-NN en géométrie cosinus : complète euclidienne/Manhattan ---
    searches["K-NN (cosinus)"] = GridSearchCV(
        Pipeline([
            ("prep", build_preprocessor(scale=True)),
            ("clf", KNeighborsClassifier(metric="cosine", weights="distance")),
        ]),
        {
            "clf__n_neighbors": [5, 11, 31, 51, 101],
        },
        **kw,
    )

    # --- 7. Extra Trees : bagging d'arbres très aléatoires ---
    searches["Extra Trees"] = RandomizedSearchCV(
        Pipeline([
            ("prep", _tree_preprocessor()),
            ("clf", ExtraTreesClassifier(class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
        {
            "clf__n_estimators": [200, 300],
            "clf__max_depth": [3, 5, 8, None],
            "clf__min_samples_leaf": [5, 10, 20],
            "clf__max_features": ["sqrt", 0.5],
        },
        n_iter=8, random_state=RANDOM_STATE, **kw,
    )

    # --- 8. AdaBoost : renforcement faible à faible (arbres peu profonds) ---
    searches["AdaBoost"] = GridSearchCV(
        Pipeline([
            ("prep", _tree_preprocessor()),
            ("clf", AdaBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=2, random_state=RANDOM_STATE),
                random_state=RANDOM_STATE,
            )),
        ]),
        {
            "clf__n_estimators": [50, 100, 200],
            "clf__learning_rate": [0.01, 0.1, 1.0],
        },
        **kw,
    )

    # --- 9. Bagging : réduction de variance autour d'un arbre équilibré ---
    searches["Bagging"] = RandomizedSearchCV(
        Pipeline([
            ("prep", _tree_preprocessor()),
            ("clf", BaggingClassifier(
                estimator=DecisionTreeClassifier(max_depth=4, class_weight="balanced",
                                                 random_state=RANDOM_STATE),
                random_state=RANDOM_STATE,
            )),
        ]),
        {
            "clf__n_estimators": [50, 100],
            "clf__max_samples": [0.5, 0.8, 1.0],
            "clf__max_features": [0.5, 1.0],
        },
        n_iter=6, random_state=RANDOM_STATE, **kw,
    )

    # --- 10. Sélection de variables DANS le Pipeline (donc dans chaque CV) ---
    searches["Sélection de variables"] = GridSearchCV(
        Pipeline([
            ("prep", build_preprocessor(scale=True)),
            ("select", SelectKBest(score_func=f_classif)),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=2000,
                                       random_state=RANDOM_STATE)),
        ]),
        {"select__k": [5, 8, 12]},
        **kw,
    )

    if XGBClassifier is not None:
        searches["XGBoost"] = GridSearchCV(
            Pipeline([
                ("prep", _tree_preprocessor()),
                ("clf", XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                )),
            ]),
            {
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [2, 3, 5],
                "clf__learning_rate": [0.03, 0.1],
                "clf__subsample": [0.8, 1.0],
            },
            **kw,
        )
    return searches


# --------------------------------------------------------------------------- #
# Ajustement avec cache (évite de relancer les recherches coûteuses)
# --------------------------------------------------------------------------- #
def fit_search(name: str, search, X, y, force: bool = False):
    """Ajuste la recherche, ou recharge le résultat mis en cache dans models/cache/."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{slug(name)}_search.joblib"
    if path.exists() and not force:
        return joblib.load(path)
    search.fit(X, y)
    joblib.dump(search, path, compress=3)
    return search


def final_estimator(name: str, search):
    """Estimateur final (non ajusté) avec des probabilités **calibrées**.

    Les poids de classes (``class_weight='balanced'``) gonflent les probabilités
    (un employé dont le risque réel est ~20 % peut recevoir 0,75), et le SVM n'a
    pas de probabilités natives. Tous les modèles sont donc enveloppés dans une
    calibration sigmoïde (Platt) en validation croisée à 5 plis. La calibration
    est monotone : elle ne modifie ni le classement des employés, ni l'AUC.
    """
    return CalibratedClassifierCV(clone(search.best_estimator_), method="sigmoid", cv=5, ensemble=False)


def cv_summary(searches: dict) -> pd.DataFrame:
    """Meilleurs hyperparamètres et scores de validation croisée (moyenne ± écart-type)."""
    rows = []
    for name, s in searches.items():
        i, r = s.best_index_, s.cv_results_
        rows.append({
            "modèle": name,
            "type": "extension" if name in EXTENSIONS else "cours",
            "recherche": type(s).__name__,
            "combinaisons": len(r["params"]),
            "AP_cv": r["mean_test_ap"][i], "AP_std": r["std_test_ap"][i],
            "ROC_AUC_cv": r["mean_test_roc_auc"][i], "F1_cv": r["mean_test_f1"][i],
            "meilleurs_hyperparamètres": {k.replace("clf__", ""): v for k, v in s.best_params_.items()},
        })
    return pd.DataFrame(rows).set_index("modèle")


# --------------------------------------------------------------------------- #
# Modèle final pour l'application
# --------------------------------------------------------------------------- #
def save_bundle(name: str, estimator, threshold: float, info: dict | None = None,
                path: Path | None = None) -> Path:
    path = path or MODELS_DIR / "best_model.joblib"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"name": name, "model": estimator, "threshold": float(threshold),
                 "features": FEATURE_COLS, "info": info or {}}, path, compress=3)
    return path


def _legacy_threshold() -> float:
    """Seuil de l'artefact historique (Pipeline nu, sans seuil optimisé)."""
    path = RESULTS_DIR / "selection_metadata.json"
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
        return float(meta.get("threshold", meta.get("best_threshold", 0.5)))
    except (OSError, TypeError, ValueError):
        return 0.5


def _as_bundle(artifact) -> dict:
    """Normalise un ancien ``best_model.joblib`` (Pipeline nu) en bundle complet."""
    if isinstance(artifact, dict) and "model" in artifact and "threshold" in artifact:
        return artifact
    return {
        "name": "legacy",
        "model": artifact,
        "threshold": _legacy_threshold(),
        "features": FEATURE_COLS,
        "info": {"legacy_artifact": True},
    }


def load_bundle(path: Path | None = None) -> dict:
    """Charge le bundle et accepte aussi les artefacts Pipeline des anciennes versions."""
    return _as_bundle(joblib.load(path or MODELS_DIR / "best_model.joblib"))


def predict_employee(bundle: dict, record: dict | pd.DataFrame) -> dict:
    """Prédit le risque de départ d'un ou plusieurs employés (clés = FEATURE_COLS).

    Renvoie la probabilité de départ, la décision au seuil optimisé et ce seuil.
    ``OverTime`` et ``PromotionLast5Years`` sont attendus en 0/1.
    """
    X = pd.DataFrame([record]) if isinstance(record, dict) else record.copy()
    X = X[FEATURE_COLS]
    p = bundle["model"].predict_proba(X)[:, 1]
    return {"probability": p, "prediction": (p >= bundle["threshold"]).astype(int),
            "threshold": bundle["threshold"]}
