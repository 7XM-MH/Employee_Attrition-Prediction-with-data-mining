"""Unit tests for model loading and prediction inference."""
import pytest
import pandas as pd
from pathlib import Path
import sys
from joblib import load

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import ALL_FEATURES, load_raw_data

MODELS_DIR = PROJECT_ROOT / "models"

# Les modèles sont des artefacts générés (voir .gitignore). Les tests ne doivent
# échouer que si l'utilisateur a oublié de lancer l'entraînement, pas à cause
# d'un checkout propre sans .joblib versionné.
pytestmark = pytest.mark.skipif(
    not (MODELS_DIR / "best_model.joblib").exists(),
    reason="Artefacts absents : lancez python scripts/train_pipeline.py",
)


def test_saved_models_exist():
    """Verify that best_model and individual pipelines are serialized."""
    assert (MODELS_DIR / "best_model.joblib").exists(), "best_model.joblib missing"
    expected_models = ["ZeroR", "KNN", "DecisionTree", "RandomForest", "NaiveBayes", "SVM"]
    for m in expected_models:
        path = MODELS_DIR / f"{m}_pipeline.joblib"
        assert path.exists(), f"Missing model pipeline {m}"


def test_best_model_inference():
    """Verify that best_model produces valid binary predictions and probabilities."""
    best_pipe = load(MODELS_DIR / "best_model.joblib")
    df = load_raw_data().head(10)
    X = df[ALL_FEATURES]
    
    preds = best_pipe.predict(X)
    assert len(preds) == 10
    assert set(preds).issubset({0, 1})
    
    if hasattr(best_pipe, "predict_proba"):
        probas = best_pipe.predict_proba(X)
        assert probas.shape == (10, 2)
        assert (probas >= 0).all() and (probas <= 1).all()
