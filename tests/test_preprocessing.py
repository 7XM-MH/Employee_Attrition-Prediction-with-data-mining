import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    FEATURE_COLS, NOMINAL_COLS, TARGET, IQRClipper,
    build_preprocessor, clean_data, split_data,
)


def make_df(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "EmployeeID": [f"E{i}" for i in range(n)],
        "Age": rng.integers(22, 60, n),
        "Department": rng.choice(["Sales", "IT", "HR"], n),
        "JobLevel": rng.integers(1, 6, n),
        "YearsAtCompany": rng.integers(0, 20, n),
        "MonthlyIncome": rng.integers(3000, 15000, n),
        "JobSatisfaction": rng.integers(1, 6, n),
        "WorkLifeBalance": rng.integers(1, 5, n),
        "OverTime": rng.choice(["Yes", "No"], n),
        "DistanceFromHome": rng.uniform(1, 30, n).round(1),
        "PromotionLast5Years": rng.choice(["Yes", "No"], n),
        "PerformanceRating": rng.integers(1, 5, n),
        "TrainingHoursLastYear": rng.integers(5, 25, n),
        "Attrition": rng.choice(["Yes", "No"], n, p=[0.1, 0.9]),
    })


def test_clean_encodes_target_and_binaries():
    df, _ = clean_data(make_df())
    assert set(df[TARGET].unique()) <= {0, 1}
    assert set(df["OverTime"].unique()) <= {0, 1}


def test_clean_removes_duplicates_and_flags_out_of_range():
    raw = make_df()
    raw = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)  # doublon exact
    raw.loc[1, "Age"] = 200                                   # aberrant
    df, rep = clean_data(raw)
    assert rep["duplicate_rows_removed"] == 1
    assert rep["out_of_range_to_nan"] == {"Age": 1}
    assert df["Age"].isna().sum() == 1


def test_clean_raises_on_missing_column():
    with pytest.raises(ValueError):
        clean_data(make_df().drop(columns=["Age"]))


def test_iqr_clipper_learns_bounds_on_fit_only():
    train = np.arange(100, dtype=float).reshape(-1, 1)
    clip = IQRClipper(factor=1.5).fit(train)
    out = clip.transform(np.array([[1e6], [-1e6], [50.0]]))
    assert out[0, 0] == clip.upper_[0] and out[1, 0] == clip.lower_[0] and out[2, 0] == 50.0


def test_preprocessor_no_leakage_and_shape():
    df, _ = clean_data(make_df())
    X_tr, X_te, _, _ = split_data(df, test_size=0.25)
    pre = build_preprocessor().fit(X_tr)
    Xtr, Xte = pre.transform(X_tr), pre.transform(X_te)
    # Moyenne ~0 sur le train pour une variable standardisée ; pas forcément sur le test
    assert abs(Xtr["Age"].mean()) < 1e-9
    assert Xtr.shape[1] == Xte.shape[1] == len(FEATURE_COLS) - len(NOMINAL_COLS) + 3
    assert not Xte.isna().any().any()


def test_unknown_category_is_ignored():
    df, _ = clean_data(make_df())
    pre = build_preprocessor().fit(df[FEATURE_COLS])
    new = df[FEATURE_COLS].head(3).copy()
    new["Department"] = "Legal"  # catégorie jamais vue
    assert pre.transform(new).filter(like="Department_").sum().sum() == 0


def test_split_is_stratified():
    df, _ = clean_data(make_df(n=1000))
    _, _, y_tr, y_te = split_data(df)
    assert abs(y_tr.mean() - y_te.mean()) < 0.02
