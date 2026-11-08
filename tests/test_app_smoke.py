"""Smoke tests for the app utilities."""
import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_utils_import():
    """Verify that key utility functions can be imported."""
    from app.utils import load_data, load_model, load_json, get_risk_level, generate_recommendations
    assert callable(load_data)


def test_risk_levels():
    """Verify risk level thresholds return correct labels."""
    from app.utils import get_risk_level
    label, css, emoji = get_risk_level(0.1)
    assert label == "Faible"
    label, css, emoji = get_risk_level(0.45)
    assert label == "Modéré"
    label, css, emoji = get_risk_level(0.8)
    assert label == "Élevé"


def test_recommendations():
    """Verify recommendations are generated for a high-risk profile."""
    from app.utils import generate_recommendations
    row = {
        "OverTime": "Yes",
        "JobSatisfaction": 1,
        "WorkLifeBalance": 1,
        "YearsAtCompany": 1,
        "MonthlyIncome": 3500,
        "DistanceFromHome": 40,
        "PromotionLast5Years": "No",
        "TrainingHoursLastYear": 5,
    }
    recs = generate_recommendations(row, 0.8)
    assert len(recs) >= 3


def test_data_loads():
    """Verify the raw dataset loads correctly if available."""
    from app.utils import load_data
    # This test depends on data file existing
    df = load_data()
    if df is not None:
        assert "Attrition" in df.columns
        assert len(df) == 10000


def test_preprocessing_import():
    """Verify preprocessing constants are importable."""
    from src.preprocessing import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET
    assert len(NUMERICAL_FEATURES) == 9
    assert len(CATEGORICAL_FEATURES) == 3
    assert TARGET == "Attrition"
