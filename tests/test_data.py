"""Validation tests for the Employee Attrition dataset.

These tests lock the characteristics of the committed raw dataset so that
any accidental modification is caught immediately.

Run with pytest:
    pytest tests/test_data.py -v

Run without pytest (fallback, no external dependency):
    python tests/test_data.py
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "employee_attrition_2026.csv"

TARGET_COLUMN = "Attrition"
ID_COLUMN = "Employee ID"

EXPECTED_SHAPE = (59_598, 24)
EXPECTED_TARGET_COUNTS = {"Stayed": 31_260, "Left": 28_338}
EXPECTED_COLUMNS = [
    "Employee ID",
    "Age",
    "Gender",
    "Years at Company",
    "Job Role",
    "Monthly Income",
    "Work-Life Balance",
    "Job Satisfaction",
    "Performance Rating",
    "Number of Promotions",
    "Overtime",
    "Distance from Home",
    "Education Level",
    "Marital Status",
    "Number of Dependents",
    "Job Level",
    "Company Size",
    "Company Tenure",
    "Remote Work",
    "Leadership Opportunities",
    "Innovation Opportunities",
    "Company Reputation",
    "Employee Recognition",
    "Attrition",
]


def load_data() -> pd.DataFrame:
    """Load the raw dataset from its committed location."""
    return pd.read_csv(DATA_PATH, encoding="utf-8")


def test_dataset_file_exists() -> None:
    assert DATA_PATH.is_file(), f"Dataset not found: {DATA_PATH}"


def test_shape() -> None:
    df = load_data()
    assert df.shape == EXPECTED_SHAPE, f"Unexpected shape: {df.shape}"


def test_expected_columns() -> None:
    df = load_data()
    assert list(df.columns) == EXPECTED_COLUMNS
    assert len(df.columns) == len(set(df.columns)), "Duplicate column names detected"


def test_no_missing_values() -> None:
    df = load_data()
    missing = int(df.isna().sum().sum())
    assert missing == 0, f"{missing} missing values detected"


def test_no_duplicate_rows() -> None:
    df = load_data()
    duplicates = int(df.duplicated().sum())
    assert duplicates == 0, f"{duplicates} duplicate rows detected"


def test_target_values() -> None:
    df = load_data()
    assert set(df[TARGET_COLUMN].unique()) == {"Stayed", "Left"}


def test_target_distribution() -> None:
    df = load_data()
    counts = df[TARGET_COLUMN].value_counts().to_dict()
    assert counts == EXPECTED_TARGET_COUNTS, f"Unexpected target counts: {counts}"


def test_employee_id_is_unique() -> None:
    df = load_data()
    assert df[ID_COLUMN].is_unique, "Employee ID must be unique"
    assert df[ID_COLUMN].notna().all(), "Employee ID must not contain missing values"


def test_feature_count() -> None:
    df = load_data()
    predictors = [c for c in df.columns if c not in (ID_COLUMN, TARGET_COLUMN)]
    assert len(predictors) == 22, f"Expected 22 predictors, found {len(predictors)}"
    assert len(predictors) >= 10, "The assignment requires at least 10 attributes"


def test_numeric_ranges() -> None:
    df = load_data()
    assert df["Age"].between(18, 59).all(), "Age outside the expected range"
    assert (df["Monthly Income"] > 0).all(), "Monthly Income must be positive"
    assert df["Years at Company"].between(1, 51).all(), "Years at Company outside the expected range"
    assert df["Number of Promotions"].between(0, 4).all()
    assert df["Number of Dependents"].between(0, 6).all()


def test_education_level_encoding() -> None:
    df = load_data()
    levels = set(df["Education Level"].unique())
    assert levels == {
        "Associate Degree",
        "Bachelor\u2019s Degree",
        "High School",
        "Master\u2019s Degree",
        "PhD",
    }


def test_company_tenure_unit_ambiguity_is_documented() -> None:
    """Guardrail: Company Tenure is very likely expressed in months.

    The raw column exceeds Age for ~71% of rows, which is only coherent if
    Company Tenure is in months while Age is in years. This test documents
    the issue instead of silently ignoring it.
    """
    df = load_data()
    inconsistency_ratio = (df["Company Tenure"] > df["Age"]).mean()
    assert inconsistency_ratio > 0.5, (
        "Company Tenure no longer exceeds Age for most rows: "
        "the dataset or the unit assumption changed, please review the data dictionary."
    )


if __name__ == "__main__":
    tests = [
        (name, obj)
        for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    failures = 0
    for name, test in tests:
        try:
            test()
            print(f"PASS {name}")
        except AssertionError as exc:  # noqa: BLE001 - simple standalone runner
            failures += 1
            print(f"FAIL {name}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} tests passed")
    raise SystemExit(1 if failures else 0)
