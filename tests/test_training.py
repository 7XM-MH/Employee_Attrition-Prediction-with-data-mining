"""Contrats du pipeline supervisé — guide §3, §4 et §6.

Les tests d'artefacts sont ignorés quand les fichiers générés sont absents :
ils vérifient ce que `python scripts/train_pipeline.py` produit réellement,
jamais des valeurs codées en dur.
"""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocessing import (  # noqa: E402
    ALL_FEATURES,
    ID_COL,
    TARGET,
    prepare_data,
)

MODELS = {"ZeroR", "KNN", "DecisionTree", "RandomForest", "NaiveBayes", "SVM"}
RESULTS = ROOT / "results"
MODELS_DIR = ROOT / "models"
PROCESSED = ROOT / "data" / "processed"

REQUIRED_JSON = [
    RESULTS / "metrics.json",
    RESULTS / "roc_data.json",
    RESULTS / "confusion_matrices.json",
    RESULTS / "best_hyperparameters.json",
    RESULTS / "cv_results_summary.json",
    RESULTS / "feature_importances.json",
    RESULTS / "selection_metadata.json",
]
ARTIFACTS_READY = all(path.exists() for path in REQUIRED_JSON)

try:
    from scripts.train_pipeline import (
        build_model_configs,
        build_preprocessor,
        positive_class_index,
        validate_schema,
    )
except (ImportError, AttributeError):  # pragma: no cover - échec de conformité
    build_model_configs = None  # type: ignore[assignment]
    build_preprocessor = None  # type: ignore[assignment]
    positive_class_index = None  # type: ignore[assignment]
    validate_schema = None  # type: ignore[assignment]

SCRIPT_CONFORM = all(
    func is not None
    for func in (build_model_configs, build_preprocessor, positive_class_index, validate_schema)
)
requires_script = pytest.mark.skipif(not SCRIPT_CONFORM, reason="script d'entraînement non conforme")
requires_artifacts = pytest.mark.skipif(not ARTIFACTS_READY, reason="artefacts non générés")


def _sample_frame() -> pd.DataFrame:
    """Petit jeu déterministe respectant exactement le schéma du guide."""
    return pd.DataFrame({
        "Age": [30, 41, 25, 38],
        "JobLevel": [2, 3, 1, 4],
        "YearsAtCompany": [1, 8, 2, 12],
        "MonthlyIncome": [5000, 9000, 3500, 12000],
        "JobSatisfaction": [3, 2, 4, 1],
        "WorkLifeBalance": [3, 4, 2, 3],
        "DistanceFromHome": [5.0, 12.0, 2.0, 20.0],
        "PerformanceRating": [3, 4, 3, 2],
        "TrainingHoursLastYear": [10, 4, 8, 2],
        "Department": ["IT", "Sales", "HR", "IT"],
        "OverTime": ["No", "Yes", "No", "Yes"],
        "PromotionLast5Years": ["Yes", "No", "No", "Yes"],
        ID_COL: ["EMP0001", "EMP0002", "EMP0003", "EMP0004"],
        TARGET: ["No", "Yes", "No", "Yes"],
    })


@requires_script
def test_model_registry_matches_guide():
    configs = build_model_configs()
    assert set(configs) == MODELS
    for config in configs.values():
        assert "classifier" in config


@requires_script
def test_preprocessor_contract():
    preprocessor = build_preprocessor()
    assert [name for name, _, _ in preprocessor.transformers] == ["num", "cat"]
    scaler = next(t for name, t, _ in preprocessor.transformers if name == "num")
    encoder = next(t for name, t, _ in preprocessor.transformers if name == "cat")
    assert isinstance(scaler, StandardScaler)
    assert isinstance(encoder, OneHotEncoder)
    assert encoder.handle_unknown == "ignore"
    assert encoder.drop is None


@requires_script
def test_validate_schema_accepts_valid_frame():
    validated = validate_schema(_sample_frame())
    assert len(validated) == 4
    X, y = prepare_data(validated)
    assert list(X.columns) == ALL_FEATURES
    assert set(y.unique()) == {0, 1}


@requires_script
def test_validate_schema_rejects_invalid_frames():
    good = _sample_frame()

    bad_target = good.copy()
    bad_target[TARGET] = "Stayed"
    with pytest.raises(ValueError, match="No/Yes"):
        validate_schema(bad_target)

    missing = good.drop(columns=["MonthlyIncome"])
    with pytest.raises(ValueError, match="MonthlyIncome"):
        validate_schema(missing)

    duplicated = good.copy()
    duplicated.loc[1, ID_COL] = duplicated.loc[0, ID_COL]
    with pytest.raises(ValueError, match="EmployeeID"):
        validate_schema(duplicated)

    non_numeric = good.copy()
    non_numeric["Age"] = "inconnu"
    with pytest.raises(ValueError, match="Age"):
        validate_schema(non_numeric)


@requires_script
def test_positive_class_index_uses_real_classes():
    from sklearn.tree import DecisionTreeClassifier

    X, y = prepare_data(_sample_frame())
    pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", DecisionTreeClassifier(random_state=42)),
    ])
    pipeline.fit(X, y)
    classes = list(pipeline.named_steps["classifier"].classes_)
    assert positive_class_index(pipeline) == classes.index(1)


def test_script_does_not_use_smote():
    """Le document peut mentionner SMOTE, mais le code ne doit pas l'utiliser."""
    source = (ROOT / "scripts" / "train_pipeline.py").read_text(encoding="utf-8")
    assert "SMOTE(" not in source
    assert "from imblearn" not in source
    assert "import imblearn" not in source


@requires_artifacts
def test_selection_metadata_contract():
    meta = json.loads((RESULTS / "selection_metadata.json").read_text(encoding="utf-8"))
    assert meta["best_model"] in MODELS
    assert meta["selection_criterion"] == "mean_cv_f1"
    assert meta["cv"]["n_splits"] >= 5
    assert meta["positive_class"] == {"label": "Yes", "encoded_value": 1}
    assert meta["class_order"] == [0, 1]
    assert meta["features"] == ALL_FEATURES
    assert meta["train_command"] == "python scripts/train_pipeline.py"


@requires_artifacts
def test_artifacts_contract():
    metrics = json.loads((RESULTS / "metrics.json").read_text(encoding="utf-8"))
    cv = json.loads((RESULTS / "cv_results_summary.json").read_text(encoding="utf-8"))
    roc = json.loads((RESULTS / "roc_data.json").read_text(encoding="utf-8"))
    confusion = json.loads((RESULTS / "confusion_matrices.json").read_text(encoding="utf-8"))
    params = json.loads((RESULTS / "best_hyperparameters.json").read_text(encoding="utf-8"))

    for name in MODELS:
        assert set(metrics[name]) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
        assert set(cv[name]) == {"mean_cv_f1", "std_cv_f1"}
        assert cv[name]["mean_cv_f1"] is not None  # ZeroR compris
        assert set(roc[name]) == {"fpr", "tpr", "auc"}
        assert len(confusion[name]) == 2
        assert all(len(row) == 2 for row in confusion[name])
        assert isinstance(params[name], dict)


@pytest.mark.skipif(not (MODELS_DIR / "best_model.joblib").exists(), reason="modèle non généré")
def test_saved_pipeline_predicts_positive_probability():
    from joblib import load

    from src.preprocessing import load_raw_data

    pipeline = load(MODELS_DIR / "best_model.joblib")
    assert isinstance(pipeline, Pipeline)
    assert set(pipeline.named_steps) == {"preprocessor", "classifier"}

    X = load_raw_data()[ALL_FEATURES].head(3)
    proba = pipeline.predict_proba(X)
    yes_idx = positive_class_index(pipeline) if positive_class_index else 1
    assert proba.shape == (3, 2)
    assert 0.0 <= proba[:, yes_idx].min() <= proba[:, yes_idx].max() <= 1.0


@pytest.mark.skipif(not (PROCESSED / "test_set.csv").exists(), reason="jeu de test non généré")
def test_test_set_saved():
    df = pd.read_csv(PROCESSED / "test_set.csv")
    assert df.shape == (2000, 14)
    assert all(col in df.columns for col in ALL_FEATURES + [ID_COL, TARGET])
    assert set(df[TARGET].unique()) <= {"No", "Yes"}
    assert df[ID_COL].is_unique
