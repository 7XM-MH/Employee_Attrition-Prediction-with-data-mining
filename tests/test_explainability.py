"""Tests de l'explicabilité locale centralisée (lot 4)."""
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.explainability import explain_local, explain_many
from src.models import save_bundle
from src.preprocessing import FEATURE_COLS, ID_COL, TARGET, build_preprocessor, clean_data
from tests.test_preprocessing import make_df


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    df, _ = clean_data(make_df(n=300, seed=17))
    X, y = df[FEATURE_COLS], df[TARGET]
    pipe = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", DecisionTreeClassifier(max_depth=3, random_state=42)),
    ]).fit(X, y)
    path = save_bundle("test-tree", pipe, 0.3, path=tmp_path_factory.mktemp("bundle") / "b.joblib")
    from src.models import load_bundle
    return load_bundle(path), df


def test_explanation_has_three_original_features(bundle):
    model, df = bundle
    out = explain_local(model, df[FEATURE_COLS].iloc[0].to_dict(), top_k=3)
    assert len(out["top_factors"]) == 3
    assert {f["feature"] for f in out["top_factors"]} <= set(FEATURE_COLS)
    assert out["explainer"] == "TreeExplainer"
    assert 0 <= out["probability"] <= 1
    assert out["prediction"] in (0, 1)


def test_explanation_is_deterministic(bundle):
    model, df = bundle
    record = df[FEATURE_COLS].iloc[1].to_dict()
    a = explain_local(model, record)
    b = explain_local(model, record)
    assert a == b


def test_direction_matches_sign_of_contribution(bundle):
    model, df = bundle
    out = explain_local(model, df[FEATURE_COLS].iloc[2].to_dict())
    for factor in out["top_factors"]:
        expected = "augmente_le_risque" if factor["contribution"] > 0 else "diminue_le_risque"
        assert factor["impact"] == expected


def test_target_and_id_are_rejected(bundle):
    model, df = bundle
    record = df[FEATURE_COLS].iloc[0].to_dict()
    record[TARGET] = 1
    with pytest.raises(ValueError):
        explain_local(model, record)
    record.pop(TARGET)
    record[ID_COL] = "EMP-0001"
    with pytest.raises(ValueError):
        explain_local(model, record)


def test_explain_many_returns_one_row_per_rank(bundle):
    model, df = bundle
    table = explain_many(model, df[FEATURE_COLS].head(4))
    assert len(table) == 12
    assert list(table.columns) == ["ligne", "rang", "feature", "contribution", "impact", "probability"]
    assert set(table["rang"]) == {1, 2, 3}
    assert set(table["feature"]) <= set(FEATURE_COLS)


def test_non_tree_model_uses_background_explainer(bundle):
    _, df = bundle
    X, y = df[FEATURE_COLS], df[TARGET]
    pipe = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", LogisticRegression(max_iter=2000)),
    ]).fit(X, y)
    # Le bundle simple peut être construit directement : aucun artefact disque n'est nécessaire.
    local_bundle = {"model": pipe, "threshold": 0.3, "features": FEATURE_COLS}
    out = explain_local(local_bundle, X.iloc[0].to_dict(), background=X.head(20))
    assert out["explainer"] != "TreeExplainer"
    assert len(out["top_factors"]) == 3
    assert np.isfinite([f["contribution"] for f in out["top_factors"]]).all()
