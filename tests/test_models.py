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
