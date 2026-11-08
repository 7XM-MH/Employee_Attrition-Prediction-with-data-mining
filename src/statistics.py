"""Statistical analysis utilities."""
import pandas as pd
import numpy as np
from scipy import stats


def compute_descriptive_stats(df: pd.DataFrame, numerical_cols: list) -> pd.DataFrame:
    """Compute descriptive statistics for numerical columns."""
    return df[numerical_cols].describe().T


def chi2_test(df: pd.DataFrame, col: str, target: str) -> dict:
    """Perform chi-squared test of independence."""
    contingency = pd.crosstab(df[col], df[target])
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    return {"chi2": round(chi2, 4), "p_value": round(p, 6), "dof": dof, "significant": p < 0.05}


def ttest_by_group(df: pd.DataFrame, col: str, target: str) -> dict:
    """Perform independent t-test between target groups."""
    group0 = df[df[target] == "No"][col].dropna()
    group1 = df[df[target] == "Yes"][col].dropna()
    t_stat, p_value = stats.ttest_ind(group0, group1)
    return {"t_statistic": round(t_stat, 4), "p_value": round(p_value, 6), "significant": p_value < 0.05}


def correlation_matrix(df: pd.DataFrame, numerical_cols: list) -> pd.DataFrame:
    """Compute correlation matrix."""
    return df[numerical_cols].corr()
