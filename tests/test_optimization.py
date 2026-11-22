import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV

from src.models import CV, FEATURE_COLS, SCORING, get_knn_pipeline
from src.optimization import (
    decision_table, net_benefit_per_1000, positive_scores, random_search_simulation, with_selector,
)
from src.preprocessing import clean_data
from tests.test_preprocessing import make_df


@pytest.fixture(scope="module")
def data():
    df, _ = clean_data(make_df(n=500, seed=4))
    return df[FEATURE_COLS], df["Attrition"]


def test_with_selector_reduces_columns_without_leak(data):
    X, y = data
    pipe = with_selector(get_knn_pipeline(), k=4).fit(X, y)
    assert len(pipe[:-1].get_feature_names_out()) == 4
    p = positive_scores(pipe, X)
    assert p.shape == (len(X),)


def test_random_search_simulation_improves_with_more_draws(data):
    X, y = data
    gs = GridSearchCV(get_knn_pipeline(), {"clf__n_neighbors": [3, 5, 9, 15, 25, 35, 51, 75]},
                      scoring=SCORING, refit="ap", cv=CV).fit(X, y)
    sim = random_search_simulation({"K-NN": gs}, n_draws=(2, 4, 6), n_sim=300)
    assert sim["AP_moyen_trouvé"].is_monotonic_increasing
    assert (sim["AP_moyen_trouvé"] <= sim["AP_optimum_grille"] + 1e-12).all()


def test_net_benefit_math():
    y = np.array([1, 0, 0, 1, 0])
    p = np.array([0.9, 0.8, 0.1, 0.7, 0.2])
    # seuil 0,5 : 3 alertes dont 2 vrais départs ; R = 4 -> (4*2 - 3)/5*1000
    assert net_benefit_per_1000(y, p, 0.5, 4) == pytest.approx((4 * 2 - 3) / 5 * 1000)
    assert net_benefit_per_1000(y, p, 1.01, 4) == 0.0


def test_decision_table_threshold_is_inverse_of_R_and_beats_alerting_everyone():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.1, 3000)
    p = np.clip(0.05 + 0.25 * y * rng.random(3000) + 0.03 * rng.random(3000), 0, 1)   # score informatif
    t = decision_table(y, p, ratios=(5, 10, 20))
    assert np.allclose(t["seuil_optimal"], [1 / 5, 1 / 10, 1 / 20])
    assert (t["bénéfice_modèle"] >= t["bénéfice_ne_rien_faire"] - 1e-9).all()
