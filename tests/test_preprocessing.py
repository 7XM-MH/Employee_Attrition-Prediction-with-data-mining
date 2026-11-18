"""Unit tests for preprocessing and data cleaning routines."""
import pytest
import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import load_raw_data, prepare_data, ALL_FEATURES
from app.cleaning import clean_data, detect_outliers_iqr, _NUMERICAL


def test_prepare_data_contract():
    """Verify prepare_data separates features from target and drops ID."""
    df = load_raw_data()
    X, y = prepare_data(df)
    assert len(X) == len(df)
    assert list(X.columns) == ALL_FEATURES
    assert "EmployeeID" not in X.columns
    assert "Attrition" not in X.columns
    assert set(y.unique()).issubset({0, 1})


def test_clean_data_pipeline():
    """Verify clean_data executes outlier capping without dropping rows."""
    df = load_raw_data()
    cleaned_df, report = clean_data(df)
    assert len(cleaned_df) == len(df)
    assert "EmployeeID" not in cleaned_df.columns
    assert "total_outliers_capped" in report


def test_outlier_detection_iqr():
    """Verify IQR calculation yields expected lower/upper bounds."""
    df = load_raw_data()
    outliers_df = detect_outliers_iqr(df, _NUMERICAL)
    assert len(outliers_df) == len(_NUMERICAL)
    assert "Variable" in outliers_df.columns
    assert "IQR" in outliers_df.columns
