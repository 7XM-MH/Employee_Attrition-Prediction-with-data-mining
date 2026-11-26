import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV

from src.models import (
    CV, EXTENSIONS, FEATURE_COLS, get_searches, get_zero_r, load_bundle, predict_employee,
    save_bundle, slug,
)
from src.preprocessing import clean_data
from tests.test_preprocessing import make_df


@pytest.fixture(scope="module")
def data():
    df, _ = clean_data(make_df(n=400, seed=3))
    return df[FEATURE_COLS], df["Attrition"]


def test_cv_is_stratified_5_fold():
    assert CV.get_n_splits() == 5


def test_all_required_algorithms_are_defined():
    names = set(get_searches())
    assert {"K-NN", "Arbre de décision", "Random Forest", "SVM"} <= names
    assert any(n.startswith("Naive Bayes") for n in names)
    assert set(EXTENSIONS) <= names


def test_every_search_uses_kfold_5_and_a_grid():
    for name, s in get_searches().items():
        assert s.cv.get_n_splits() == 5, name
        assert (s.param_grid if hasattr(s, "param_grid") else s.param_distributions), name


def test_pipelines_fit_and_give_probabilities(data):
    X, y = data
    ests = {"Zero-R": get_zero_r()}
    ests.update({n: s.estimator for n, s in get_searches().items() if n != "SVM"})
    ests["SVM"] = CalibratedClassifierCV(clone(get_searches()["SVM"].estimator), cv=3, ensemble=False)
    for name, est in ests.items():
        p = clone(est).fit(X, y).predict_proba(X)[:, 1]
        assert p.shape == (len(X),) and np.all((p >= 0) & (p <= 1)), name


def test_slug_is_filesystem_safe():
    assert slug("Naive Bayes (Gaussian)") == "naive_bayes_gaussian"


def test_bundle_roundtrip_and_prediction(data, tmp_path):
    X, y = data
    model = clone(get_searches()["Régression logistique"].estimator).fit(X, y)
    path = save_bundle("LR", model, 0.3, path=tmp_path / "b.joblib")
    bundle = load_bundle(path)
    rec = X.iloc[0].to_dict()
    out = predict_employee(bundle, rec)
    assert 0 <= out["probability"][0] <= 1 and out["prediction"][0] in (0, 1) and out["threshold"] == 0.3


def test_legacy_pipeline_bundle_is_normalised(data, tmp_path, monkeypatch):
    """An old best_model.joblib containing a bare Pipeline must stay usable."""
    import joblib
    from src import models as models_module
    from src.models import _as_bundle

    X, y = data
    model = clone(get_searches()["Régression logistique"].estimator).fit(X, y)
    path = tmp_path / "legacy.joblib"
    joblib.dump(model, path)
    # The historical guide pipeline has no optimized threshold: 0.5 is authoritative.
    monkeypatch.setattr(models_module, "RESULTS_DIR", tmp_path / "no-results")
    bundle = _as_bundle(joblib.load(path))
    assert bundle["model"] is not None
    assert bundle["threshold"] == 0.5
    assert bundle["features"] == FEATURE_COLS
    out = predict_employee(bundle, X.iloc[0].to_dict())
    assert out["prediction"][0] in (0, 1) and out["threshold"] == 0.5


def test_predict_employee_ignores_ids_and_target(data):
    """EmployeeID and Attrition must never reach the model."""
    X, y = data
    model = clone(get_searches()["Régression logistique"].estimator).fit(X, y)
    bundle = {"model": model, "threshold": 0.5}
    rec = X.iloc[0].to_dict()
    expected = predict_employee(bundle, rec)
    polluted = dict(rec)
    polluted["EmployeeID"] = "EMP0001"
    polluted["Attrition"] = 1
    actual = predict_employee(bundle, polluted)
    assert list(actual["probability"]) == list(expected["probability"])
    assert list(actual["prediction"]) == list(expected["prediction"])


def test_predict_employee_requires_all_model_features(data):
    """A missing model feature must fail loudly instead of being guessed."""
    X, y = data
    model = clone(get_searches()["Régression logistique"].estimator).fit(X, y)
    bundle = {"model": model, "threshold": 0.5}
    incomplete = X.iloc[0].to_dict()
    incomplete.pop(FEATURE_COLS[0])
    with pytest.raises(KeyError):
        predict_employee(bundle, incomplete)
