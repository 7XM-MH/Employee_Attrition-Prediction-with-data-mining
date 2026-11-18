"""Unit tests for dataset integrity and validation."""
import pytest
import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import DATA_PATH, ALL_FEATURES, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET, ID_COL


def test_raw_dataset_exists():
    """Verify that the raw CSV file exists and is accessible."""
    assert DATA_PATH.exists(), f"Missing dataset file at {DATA_PATH}"


def test_dataset_dimensions():
    """Verify minimum requirement of >= 10,000 rows and expected schema."""
    df = pd.read_csv(DATA_PATH)
    assert len(df) >= 10000, f"Expected >= 10,000 rows, found {len(df)}"
    assert TARGET in df.columns, f"Target column '{TARGET}' missing"
    assert ID_COL in df.columns, f"Identifier column '{ID_COL}' missing"
    for col in ALL_FEATURES:
        assert col in df.columns, f"Feature '{col}' missing from raw data"


def test_no_null_values():
    """Verify there are no missing values in raw dataset."""
    df = pd.read_csv(DATA_PATH)
    assert df.isnull().sum().sum() == 0, "Raw dataset contains null values"


def test_target_distribution():
    """Verify binary target modalities and class imbalance."""
    df = pd.read_csv(DATA_PATH)
    counts = df[TARGET].value_counts()
    assert "No" in counts and "Yes" in counts
    # Class imbalance: positive class Yes should be ~9%
    attrition_rate = counts["Yes"] / len(df)
    assert 0.05 <= attrition_rate <= 0.15, f"Unexpected attrition rate {attrition_rate}"
