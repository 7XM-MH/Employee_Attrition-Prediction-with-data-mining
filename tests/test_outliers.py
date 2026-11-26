"""Tests déterministes du module d'analyse des outliers (LOT 2)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_outliers import CSV_RELATIVE_PATH, FIGURE_RELATIVE_PATH, run
from src.outliers import (
    SUMMARY_COLUMNS,
    iqr_outliers,
    isolation_forest_outliers,
    lof_outliers,
    outlier_summary,
    zscore_outliers,
)
from src.preprocessing import ID_COL, NUMERIC_COLS, ORDINAL_COLS, TARGET, iqr_outlier_report

COLS = list(NUMERIC_COLS) + list(ORDINAL_COLS)


def make_df(n: int = 60, seed: int = 7) -> pd.DataFrame:
    """Petit jeu synthétique avec le schéma complet attendu par le projet."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        ID_COL: [f"EMP{i:04d}" for i in range(n)],
        "Age": rng.integers(25, 55, n),
        "YearsAtCompany": rng.integers(0, 20, n),
        "MonthlyIncome": rng.integers(3000, 15000, n),
        "DistanceFromHome": rng.uniform(1, 30, n).round(1),
        "TrainingHoursLastYear": rng.integers(2, 31, n),
        "JobLevel": rng.integers(1, 6, n),
        "JobSatisfaction": rng.integers(1, 6, n),
        "WorkLifeBalance": rng.integers(1, 5, n),
        "PerformanceRating": rng.integers(1, 5, n),
        TARGET: rng.choice(["No", "Yes"], n, p=[0.9, 0.1]),
    })


def test_zscore_detects_known_extreme_value_only():
    df = make_df(n=21)
    df.loc[0, "MonthlyIncome"] = 5_000_000
    rapport = zscore_outliers(df, ["MonthlyIncome"])

    assert len(rapport) == 1
    assert rapport.loc[0, "variable"] == "MonthlyIncome"
    assert rapport.loc[0, "n_outliers"] == 1
    assert rapport.loc[0, "pct_outliers"] == round(100 / 21, 2)
    assert rapport.loc[0, "seuil"] == "|z| > 3"


def test_iqr_delegates_to_preprocessing_report():
    df = make_df()
    ours = iqr_outliers(df, COLS)
    reference = iqr_outlier_report(df, cols=COLS, factors=(1.5,))

    assert list(ours["variable"]) == list(reference.index)
    assert ours["n_outliers"].tolist() == reference["outliers_k=1.5"].tolist()
    assert (ours["seuil"] == "IQR x1.5").all()


def test_multivariate_methods_return_correct_lengths_and_are_reproducible():
    X = make_df()[COLS]

    first_if = isolation_forest_outliers(X, random_state=42)
    second_if = isolation_forest_outliers(X, random_state=42)
    first_lof = lof_outliers(X, n_neighbors=20)
    second_lof = lof_outliers(X, n_neighbors=20)

    assert len(first_if) == len(first_lof) == len(X)
    assert first_if.dtype == bool and first_lof.dtype == bool
    pd.testing.assert_series_equal(first_if, second_if)
    pd.testing.assert_series_equal(first_lof, second_lof)


def test_summary_schema_and_no_target_or_identifier():
    df = make_df()
    summary = outlier_summary(df)

    assert list(summary.columns) == SUMMARY_COLUMNS
    assert set(summary["methode"]) == {"Z-score", "IQR", "Isolation Forest", "LOF"}
    assert ID_COL not in set(summary["variable"])
    assert TARGET not in set(summary["variable"])
    assert (summary["n_outliers"] >= 0).all()
    assert (summary["pct_outliers"] >= 0).all() and (summary["pct_outliers"] <= 100).all()
    assert summary["action_recommandee"].str.contains("vérification|erreur avérée").all()


def test_cli_produces_readable_and_consistent_artifacts(tmp_path: Path):
    raw_dir = tmp_path / "data/raw"
    raw_dir.mkdir(parents=True)
    make_df(n=40).to_csv(raw_dir / "Employee_Attrition_DataSet.csv", index=False)

    summary = run(tmp_path)
    csv_path = tmp_path / CSV_RELATIVE_PATH
    figure_path = tmp_path / FIGURE_RELATIVE_PATH

    assert csv_path.exists() and csv_path.stat().st_size > 0
    assert figure_path.exists() and figure_path.stat().st_size > 0
    reloaded = pd.read_csv(csv_path)
    assert list(reloaded.columns) == SUMMARY_COLUMNS
    assert len(reloaded) == len(summary) == 2 * len(COLS) + 2 == 20
    expected = (100 * reloaded["n_outliers"] / 40).round(2)
    pd.testing.assert_series_equal(
        reloaded["pct_outliers"], expected, check_names=False
    )


def test_no_direct_deletion_is_recommended():
    df = make_df()
    summary = outlier_summary(df)
    multivariate = summary[summary["methode"].isin(["Isolation Forest", "LOF"])]

    assert len(multivariate) == 2
    assert multivariate["action_recommandee"].str.contains("suppression uniquement").all()
    assert any("winsorisation" in action for action in summary["action_recommandee"])
