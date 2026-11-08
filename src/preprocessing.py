"""Data preprocessing utilities."""
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Employee_Attrition_DataSet.csv"

NUMERICAL_FEATURES = [
    "Age", "JobLevel", "YearsAtCompany", "MonthlyIncome",
    "JobSatisfaction", "WorkLifeBalance", "DistanceFromHome",
    "PerformanceRating", "TrainingHoursLastYear"
]

CATEGORICAL_FEATURES = ["Department", "OverTime", "PromotionLast5Years"]

TARGET = "Attrition"
ID_COL = "EmployeeID"

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


def load_raw_data() -> pd.DataFrame:
    """Load the raw dataset."""
    return pd.read_csv(DATA_PATH)


def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Drop ID, encode target, return X and y."""
    df = df.copy()
    y = df[TARGET].map({"Yes": 1, "No": 0})
    X = df[ALL_FEATURES]
    return X, y
