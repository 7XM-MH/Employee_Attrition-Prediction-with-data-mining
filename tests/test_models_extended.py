"""Couverture du lot 4 : modèles supplémentaires et contrat du pipeline du guide."""
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.neighbors import KNeighborsClassifier

from scripts.train_pipeline import ALL_FEATURES, build_model_configs, prepare_data
from src.models import CV, EXTENSIONS, FEATURE_COLS, get_searches
from src.preprocessing import TARGET, clean_data
from tests.test_preprocessing import make_df


@pytest.fixture(scope="module")
def data():
    df, _ = clean_data(make_df(n=300, seed=11))
    return df[FEATURE_COLS], df[TARGET]


REQUIRED_GUIDE_NAMES = ["ZeroR", "KNN", "DecisionTree", "RandomForest", "NaiveBayes", "SVM"]
NEW_NAMES = [
    "Extra Trees",
    "AdaBoost",
    "Bagging",
    "K-NN (cosinus)",
    "Sélection de variables",
]


def test_train_pipeline_imports_and_keeps_the_six_required_models():
    configs = build_model_configs()
    assert list(configs)[:6] == REQUIRED_GUIDE_NAMES
    assert len(ALL_FEATURES) == 12
    assert "EmployeeID" not in ALL_FEATURES


def test_prepare_data_encodes_target_without_an_extra_schema():
    raw = make_df(n=40, seed=5)
    X, y = prepare_data(raw)
    assert list(X.columns) == ALL_FEATURES
    assert set(y.unique()) <= {0, 1}


def test_new_models_are_registered_as_extensions():
    names = set(get_searches())
    for name in NEW_NAMES:
        assert name in names
        assert name in EXTENSIONS


def test_cosine_knn_uses_the_cosine_metric():
    est = get_searches()["K-NN (cosinus)"].estimator
    assert isinstance(est.named_steps["clf"], KNeighborsClassifier)
    assert est.named_steps["clf"].metric == "cosine"


def test_feature_selection_happens_inside_the_pipeline():
    est = get_searches()["Sélection de variables"].estimator
    select = est.named_steps["select"]
    assert isinstance(select, SelectKBest)
    ks = get_searches()["Sélection de variables"].param_grid["select__k"]
    assert 1 <= min(ks) <= max(ks) <= len(FEATURE_COLS)


def test_ensemble_estimators_are_tree_based():
    searches = get_searches()
    assert isinstance(searches["Extra Trees"].estimator.named_steps["clf"], ExtraTreesClassifier)
    assert isinstance(searches["AdaBoost"].estimator.named_steps["clf"], AdaBoostClassifier)
    assert isinstance(searches["Bagging"].estimator.named_steps["clf"], BaggingClassifier)


def test_new_pipelines_fit_and_return_positive_class_probabilities(data):
    X, y = data
    for name in NEW_NAMES:
        est = clone(get_searches()[name].estimator).fit(X, y)
        p = est.predict_proba(X)[:, 1]
        assert p.shape == (len(X),), name
        assert np.all((p >= 0) & (p <= 1)), name


def test_all_new_searches_keep_the_cv_contract():
    for name in NEW_NAMES:
        search = get_searches()[name]
        assert search.cv.get_n_splits() == CV.get_n_splits() == 5, name
        assert set(search.scoring) == {"ap", "roc_auc", "f1"}, name
