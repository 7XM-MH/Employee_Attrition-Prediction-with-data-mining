"""Regression tests for the audit fixes on the Streamlit application."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _source(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def _decision_label(res: dict) -> str:
    """Mirror the corrected page 2 decision rendering from the stored result."""
    decision = int(res["prediction"])
    return "Risque Départ" if decision else "Fidélisé"


def test_binary_decision_label_uses_prediction_not_string():
    """The binary decision must drive the label, not a stale 'Yes' string."""
    source = _source("app/pages/2_Prediction.py")
    assert 'int(res["prediction"])' in source
    assert 'if decision else "Fidélisé"' in source
    assert "pred_label" not in source
    assert _decision_label({"prediction": 1, "label": "Yes"}) == "Risque Départ"
    assert _decision_label({"prediction": 0, "label": "No"}) == "Fidélisé"


def test_feature_importance_filters_empty_models():
    """ZeroR has no usable importance and must not appear in the selector."""
    source = _source("app/pages/4_Facteurs_Demission.py")
    assert "len(values) >= 2" in source
    assert "usable_importances = {" in source
    assert "fi = usable_importances[selected_model]" in source
    assert "Gini ou permutation" in source


def test_batch_progress_column_is_percentage_scale():
    """ProgressColumn uses 0-100 for values expressed in percent."""
    source = _source("app/pages/5_Prediction_en_lot.py")
    assert '"Probabilité Risque (%)"' in source
    assert "max_value=100.0" in source


def test_exports_exclude_sensitive_columns():
    """UI exports must not leak EmployeeID or Attrition."""
    for relative in ("app/pages/1_Dashboard.py", "app/pages/5_Prediction_en_lot.py"):
        source = _source(relative)
        assert '"EmployeeID", "Attrition"' in source
        assert "exclut volontairement" in source
