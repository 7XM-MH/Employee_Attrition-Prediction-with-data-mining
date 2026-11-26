"""Explicabilité locale centralisée des prédictions d'attrition.

Le module reçoit un bundle produit par :func:`src.models.save_bundle` et
explique **une prédiction individuelle** avec SHAP. Les contributions des
variables one-hot sont recombinées pour revenir aux 12 variables métier de
:data:`src.preprocessing.FEATURE_COLS` : l'utilisateur voit donc
``Department`` et non ``Department_IT``.

Deux moteurs sont utilisés, sans dépendance supplémentaire :

* ``shap.TreeExplainer`` pour les modèles à base d'arbres (pas de fond requis) ;
* ``shap.Explainer`` avec un petit échantillon de fond pour les autres modèles.

Aucune cible ni identifiant n'est utilisé pour produire l'explication.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.tree import DecisionTreeClassifier

from .preprocessing import FEATURE_COLS, ID_COL, TARGET

DEFAULT_TOP_K = 3
BACKGROUND_SIZE = 100
RANDOM_STATE = 42

# Les modèles à base d'arbres n'ont pas besoin d'échantillon de fond.
TREE_TYPES = (
    DecisionTreeClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)


def _as_frame(record: dict | pd.Series | pd.DataFrame) -> pd.DataFrame:
    """Accepte un dict, une Series ou un DataFrame comme enregistrement unique."""
    if isinstance(record, pd.Series):
        return record.to_frame().T
    if isinstance(record, pd.DataFrame):
        return record.copy()
    return pd.DataFrame([record])


def _preprocessor_of(pipe):
    """Retourne le ColumnTransformer d'un Pipeline, quel que soit son nom d'étape."""
    for key in ("preprocessor", "prep"):
        if key in pipe.named_steps:
            return pipe.named_steps[key]
    raise ValueError("Le Pipeline ne contient pas de préprocesseur nommé 'preprocessor' ou 'prep'.")


def _classifier_of(pipe):
    for key in ("classifier", "clf"):
        if key in pipe.named_steps:
            return pipe.named_steps[key]
    raise ValueError("Le Pipeline ne contient pas de classifieur nommé 'classifier' ou 'clf'.")


def _to_frame(values, columns: list[str], index) -> pd.DataFrame:
    frame = pd.DataFrame(values, columns=list(columns))
    if index is not None and len(index) == len(frame):
        frame.index = index
    return frame


def transform_record(bundle: dict, record: dict | pd.Series | pd.DataFrame) -> pd.DataFrame:
    """Applique UNIQUEMENT le préprocesseur du Pipeline à un enregistrement."""
    X = _as_frame(record)
    X = X[FEATURE_COLS]
    prep = _preprocessor_of(bundle["model"])
    Z = prep.transform(X)
    names = [str(n) for n in prep.get_feature_names_out()]
    return _to_frame(Z, names, X.index)


def _original_name(column: str) -> str:
    """Ramène ``cat__Department_IT`` ou ``Department_IT`` à ``Department``."""
    name = str(column).split("__")[-1]
    for feature in FEATURE_COLS:
        if name == feature or name.startswith(f"{feature}_"):
            return feature
    return name


def _shap_values(explainer, Z: pd.DataFrame, is_tree: bool):
    """Calcule les valeurs SHAP et normalise les formes renvoyées par shap."""
    if is_tree:
        raw = explainer.shap_values(Z, check_additivity=False)
    else:
        raw = explainer(Z, check_additivity=False).values

    values = np.asarray(raw, dtype=float)
    if values.ndim == 3:  # (n_lignes, n_variables, n_classes)
        positive = 1 if values.shape[-1] == 2 else 0
        values = values[..., positive]
    if values.ndim == 2 and values.shape[0] == 1:
        values = values[0]
    if values.ndim != 1 or len(values) != Z.shape[1]:
        raise ValueError(f"Forme de valeurs SHAP inattendue : {values.shape}")
    return values


def explain_local(
    bundle: dict,
    record: dict | pd.Series | pd.DataFrame,
    background: pd.DataFrame | None = None,
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, Any]:
    """Explique une prédiction individuelle et retourne les ``top_k`` facteurs.

    Parameters
    ----------
    bundle : dictionnaire issu de ``src.models.save_bundle``.
    record : ligne brute avec les 12 colonnes de ``FEATURE_COLS``.
    background : échantillon brut optionnel, requis pour les modèles non-arbres.
    top_k : nombre de facteurs à retourner (3 par défaut).
    """
    if ID_COL in record or TARGET in record:
        raise ValueError("L'explication n'accepte ni EmployeeID ni Attrition.")

    X = _as_frame(record)
    X = X[FEATURE_COLS]
    Z = transform_record(bundle, X)
    clf = _classifier_of(bundle["model"])

    is_tree = isinstance(clf, TREE_TYPES)
    if background is not None and not is_tree:
        Z_bg = transform_record(bundle, background[FEATURE_COLS])
        if len(Z_bg) > BACKGROUND_SIZE:
            rng = np.random.default_rng(RANDOM_STATE)
            idx = rng.choice(len(Z_bg), size=BACKGROUND_SIZE, replace=False)
            Z_bg = Z_bg.iloc[idx]
        explainer = shap.Explainer(clf, Z_bg)
    elif is_tree:
        explainer = shap.TreeExplainer(clf)
    else:
        raise ValueError(
            "Ce modèle nécessite un échantillon de fond (argument 'background')."
        )

    values = _shap_values(explainer, Z, is_tree)
    contributions: dict[str, float] = {}
    for column, value in zip(Z.columns, values):
        name = _original_name(str(column))
        contributions[name] = contributions.get(name, 0.0) + float(value)

    ordered = sorted(contributions.items(), key=lambda item: abs(item[1]), reverse=True)[:top_k]
    classes = list(getattr(clf, "classes_", [0, 1]))
    positive = classes.index(1) if 1 in classes else 1
    p = float(bundle["model"].predict_proba(X)[0, positive])

    return {
        "probability": round(p, 4),
        "threshold": float(bundle["threshold"]),
        "prediction": int(p >= float(bundle["threshold"])),
        "explainer": type(explainer).__name__,
        "top_factors": [
            {
                "feature": name,
                "contribution": round(value, 4),
                "impact": "augmente_le_risque" if value > 0 else "diminue_le_risque",
            }
            for name, value in ordered
        ],
    }


def explain_many(
    bundle: dict,
    records: pd.DataFrame,
    background: pd.DataFrame | None = None,
    top_k: int = DEFAULT_TOP_K,
) -> pd.DataFrame:
    """Tableau aplati ``ligne, rang, feature, contribution, impact`` pour un CSV."""
    rows = []
    for position, (_, row) in enumerate(records[FEATURE_COLS].iterrows()):
        explanation = explain_local(bundle, row, background=background, top_k=top_k)
        for rank, factor in enumerate(explanation["top_factors"], start=1):
            rows.append({
                "ligne": position,
                "rang": rank,
                "feature": factor["feature"],
                "contribution": factor["contribution"],
                "impact": factor["impact"],
                "probability": explanation["probability"],
            })
    return pd.DataFrame(rows)
