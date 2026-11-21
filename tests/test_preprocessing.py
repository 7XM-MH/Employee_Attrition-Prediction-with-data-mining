"""Unit tests for preprocessing and data preparation routines."""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import (
    load_raw_data,
    prepare_data,
    ALL_FEATURES,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET,
    ID_COL,
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def test_prepare_data_contract():
    """Verify prepare_data separates features from target and drops ID."""
    df = load_raw_data()
    X, y = prepare_data(df)
    assert len(X) == len(df)
    assert list(X.columns) == ALL_FEATURES
    assert ID_COL not in X.columns
    assert TARGET not in X.columns
    assert set(y.unique()).issubset({0, 1})


def test_feature_definitions():
    """Verify feature partitions match expected counts."""
    assert len(NUMERICAL_FEATURES) == 9
    assert len(CATEGORICAL_FEATURES) == 3
    assert len(ALL_FEATURES) == 12
    assert set(ALL_FEATURES) == set(NUMERICAL_FEATURES + CATEGORICAL_FEATURES)


def test_preprocessor_transformation():
    """Verify ColumnTransformer transforms features to valid numeric matrix."""
    df = load_raw_data().head(50)
    X, y = prepare_data(df)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    
    transformed = preprocessor.fit_transform(X)
    assert transformed.shape[0] == len(df)
    assert transformed.shape[1] >= len(ALL_FEATURES)
    assert np.isfinite(transformed).all()


def test_target_mapping_is_no_zero_yes_one():
    """La cible suit exactement le contrat No -> 0 et Yes -> 1."""
    df = load_raw_data().head(50)
    _, y = prepare_data(df)
    expected = df[TARGET].map({"No": 0, "Yes": 1})
    assert y.tolist() == expected.tolist()
    assert set(y.unique()).issubset({0, 1})


def test_prepare_data_does_not_mutate_input():
    """prepare_data travaille sur une copie et n'ajoute aucune colonne."""
    df = load_raw_data().head(5)
    original = df.copy()
    X, y = prepare_data(df)
    pd.testing.assert_frame_equal(df, original)
    assert ID_COL not in X.columns
    assert TARGET not in X.columns
    assert len(y) == len(df)


def test_feature_names_are_unique():
    assert len(ALL_FEATURES) == 12
    assert len(set(ALL_FEATURES)) == 12
    assert set(ALL_FEATURES) == set(NUMERICAL_FEATURES + CATEGORICAL_FEATURES)
