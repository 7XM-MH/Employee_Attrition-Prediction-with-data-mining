"""Analyses d'optimisation (complètent le réglage fait dans ``src/models.py``).

1. ``default_vs_tuned``       : gain apporté par le réglage des hyperparamètres.
2. ``random_search_simulation``: Grid vs Random search, simulé sur les scores déjà calculés.
3. ``nested_cv``              : validation croisée imbriquée (estimation non biaisée de la
                                procédure « réglage + évaluation »).
4. ``feature_selection_study``: sélection de variables réglée *dans* la validation croisée.
5. ``decision_table``         : choix du seuil de décision selon le coût d'une intervention.
"""
from __future__ import annotations

import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from src.evaluation import paired_bootstrap_diff
from src.models import CV, RANDOM_STATE, ROOT, SCORING

OPT_DIR = ROOT / "results" / "optimization"


def positive_scores(model, X) -> np.ndarray:
    """Score de la classe positive : probabilité si disponible, sinon fonction de décision."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def _fmt(params: dict) -> str:
    return ", ".join(f"{k.replace('clf__', '').replace('prep__num__kbins__', '')}={v}" for k, v in params.items())


# --------------------------------------------------------------------------- #
# 1. Hyperparamètres par défaut vs optimisés
# --------------------------------------------------------------------------- #
def default_vs_tuned(searches: dict, X_tr, y_tr, X_te, y_te) -> pd.DataFrame:
    """Compare, à plis identiques, le modèle aux hyperparamètres par défaut et le modèle optimisé.

    Les choix non réglés (poids de classes, prétraitement) sont les mêmes dans les
    deux cas : seule l'optimisation des hyperparamètres diffère.
    """
    rows = []
    for name, s in searches.items():
        base = clone(s.estimator)
        cvr = cross_validate(base, X_tr, y_tr, cv=CV, scoring=SCORING)
        default_model = clone(base).fit(X_tr, y_tr)
        sd, st = positive_scores(default_model, X_te), positive_scores(s.best_estimator_, X_te)
        rows.append({
            "modèle": name,
            "AP_cv_défaut": cvr["test_ap"].mean(), "AP_cv_défaut_std": cvr["test_ap"].std(),
            "AP_cv_optimisé": s.best_score_,
            "AP_cv_optimisé_std": s.cv_results_["std_test_ap"][s.best_index_],
            "ROC_cv_défaut": cvr["test_roc_auc"].mean(), "ROC_cv_optimisé": s.cv_results_["mean_test_roc_auc"][s.best_index_],
            "AP_test_défaut": average_precision_score(y_te, sd), "AP_test_optimisé": average_precision_score(y_te, st),
            "ROC_test_défaut": roc_auc_score(y_te, sd), "ROC_test_optimisé": roc_auc_score(y_te, st),
            "hyperparamètres_optimaux": _fmt(s.best_params_),
        })
    out = pd.DataFrame(rows).set_index("modèle")
    out["gain_AP_cv"] = out["AP_cv_optimisé"] - out["AP_cv_défaut"]
    out["gain_AP_test"] = out["AP_test_optimisé"] - out["AP_test_défaut"]
    return out


# --------------------------------------------------------------------------- #
# 2. Grid vs Random search (simulation sans nouveau calcul)
# --------------------------------------------------------------------------- #
def random_search_simulation(searches: dict, n_draws=(3, 5, 10, 20), n_sim: int = 5000,
                             tol: float = 0.002, seed: int = 0) -> pd.DataFrame:
    """Que donnerait une recherche aléatoire de ``n`` tirages (sans remise) dans la même grille ?

    Utilise les scores déjà calculés par la grille complète : aucun réentraînement.
    ``p_proche`` = probabilité de trouver une configuration à moins de ``tol`` de l'optimum.
    La Random Forest (déjà échantillonnée) est exclue.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for name, s in searches.items():
        if type(s).__name__ != "GridSearchCV":
            continue
        scores = np.asarray(s.cv_results_["mean_test_ap"])
        n_cfg, best = len(scores), scores.max()
        grid_seconds = float((np.sum(s.cv_results_["mean_fit_time"]) + np.sum(s.cv_results_["mean_score_time"])) * s.cv.get_n_splits())
        for n in n_draws:
            if n >= n_cfg:
                continue
            draws = np.array([scores[rng.choice(n_cfg, n, replace=False)].max() for _ in range(n_sim)])
            rows.append({
                "modèle": name, "configurations_grille": n_cfg, "tirages": n,
                "part_du_calcul": n / n_cfg, "AP_moyen_trouvé": draws.mean(), "AP_optimum_grille": best,
                "perte_moyenne": best - draws.mean(), "p_proche": float((draws >= best - tol).mean()),
                "secondes_grille_complète": grid_seconds,
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 3. Validation croisée imbriquée
# --------------------------------------------------------------------------- #
def nested_cv(name: str, search, X, y, inner_splits: int = 3, outer_seed: int = 123) -> dict:
    """Validation croisée imbriquée : boucle externe 5 plis, réglage dans la boucle interne.

    Le réglage ne voit jamais le pli externe de test : le score obtenu est une
    estimation non biaisée de la performance d'un modèle **réglé**. La boucle
    interne n'utilise que 3 plis (coût de calcul).
    """
    s = clone(search)
    s.set_params(cv=StratifiedKFold(inner_splits, shuffle=True, random_state=RANDOM_STATE))
    outer = StratifiedKFold(5, shuffle=True, random_state=outer_seed)
    t = time.time()
    r = cross_validate(s, X, y, cv=outer, scoring={"ap": "average_precision", "roc_auc": "roc_auc"})
    return {"modèle": name, "AP_imbriqué": r["test_ap"].mean(), "AP_imbriqué_std": r["test_ap"].std(),
            "ROC_imbriqué": r["test_roc_auc"].mean(), "secondes": time.time() - t}


def run_nested(searches_template: dict, searches_fitted: dict, X, y, path: Path, names=None, verbose=True) -> pd.DataFrame:
    """Exécute la validation imbriquée modèle par modèle, avec reprise (fichier CSV incrémental)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    done = pd.read_csv(path, index_col=0) if path.exists() else pd.DataFrame()
    rows = done.reset_index().to_dict("records") if len(done) else []
    for name in (names or list(searches_template)):
        if any(r["modèle"] == name for r in rows):
            continue
        r = nested_cv(name, searches_template[name], X, y)
        r["AP_cv_non_imbriqué"] = searches_fitted[name].best_score_
        r["optimisme"] = r["AP_cv_non_imbriqué"] - r["AP_imbriqué"]
        rows.append(r)
        pd.DataFrame(rows).set_index("modèle").to_csv(path)
        if verbose:
            print(f"  imbriqué {name:<26} AP={r['AP_imbriqué']:.4f}  (non imbriqué {r['AP_cv_non_imbriqué']:.4f})  [{r['secondes']:.0f}s]", flush=True)
    return pd.DataFrame(rows).set_index("modèle")


# --------------------------------------------------------------------------- #
# 4. Sélection de variables réglée dans la validation croisée
# --------------------------------------------------------------------------- #
CRITERIA = {"F (ANOVA)": f_classif, "information mutuelle": partial(mutual_info_classif, random_state=0)}


def with_selector(pipe: Pipeline, k: int, criterion: str = "F (ANOVA)") -> Pipeline:
    """Insère une sélection des k meilleures colonnes après le prétraitement.

    Placée dans le Pipeline, la sélection est réapprise à chaque pli : pas de fuite.
    Critères : F de l'ANOVA (rapide, stable) ou information mutuelle (non paramétrique,
    mais estimation bruitée quand le signal est faible).
    """
    steps = list(pipe.steps)
    steps.insert(1, ("select", SelectKBest(CRITERIA[criterion], k=k)))
    return Pipeline(steps)


def feature_selection_study(searches: dict, X_tr, y_tr, X_te, y_te,
                            models=("K-NN", "Naive Bayes (Gaussian)", "SVM", "Régression logistique"),
                            ks=(3, 4, 5, 6, 8, 10, 12, 18), criteria=tuple(CRITERIA), verbose=True):
    """Pour chaque modèle (hyperparamètres déjà optimisés) et chaque critère, règle le nombre k de variables.

    Règle de l'écart-type : on retient le plus petit k dont l'AP en CV est à moins
    d'un écart-type du meilleur (modèle le plus simple statistiquement équivalent).
    """
    curve, summary = [], []
    for name in models:
        base = clone(searches[name].best_estimator_)
        s_ref = positive_scores(searches[name].best_estimator_, X_te)
        for crit in criteria:
            res = {}
            for k in ks:
                r = cross_validate(with_selector(base, k, crit), X_tr, y_tr, cv=CV, scoring={"ap": "average_precision"})
                res[k] = (r["test_ap"].mean(), r["test_ap"].std())
                curve.append({"modèle": name, "critère": crit, "k": k, "AP_cv": res[k][0], "AP_cv_std": res[k][1]})
            best_k = max(res, key=lambda k: res[k][0])
            k_sel = min(k for k in ks if res[k][0] >= res[best_k][0] - res[best_k][1])

            sel = clone(with_selector(base, k_sel, crit)).fit(X_tr, y_tr)
            s_sel = positive_scores(sel, X_te)
            diff = paired_bootstrap_diff(y_te, s_sel, s_ref, n_boot=1000)
            summary.append({
                "modèle": name, "critère": crit, "k_retenu": k_sel,
                "AP_cv_sans_sélection": res[max(ks)][0], "AP_cv_avec_sélection": res[k_sel][0],
                "AP_test_sans": average_precision_score(y_te, s_ref), "AP_test_avec": average_precision_score(y_te, s_sel),
                "ROC_test_sans": roc_auc_score(y_te, s_ref), "ROC_test_avec": roc_auc_score(y_te, s_sel),
                "ΔROC_test": diff["diff"], "ΔROC_ic_bas": diff["ic_low"], "ΔROC_ic_haut": diff["ic_high"], "ΔROC_p": diff["p_value"],
                "variables_retenues": ", ".join(sel[:-1].get_feature_names_out()),
            })
            if verbose:
                print(f"  sélection {name:<24} {crit:<21} k={k_sel:<2} AP_cv {res[max(ks)][0]:.4f} -> {res[k_sel][0]:.4f}", flush=True)
    return pd.DataFrame(curve), pd.DataFrame(summary).set_index(["modèle", "critère"])


# --------------------------------------------------------------------------- #
# 5. Optimisation de la décision : coût d'une intervention de rétention
# --------------------------------------------------------------------------- #
def net_benefit_per_1000(y, p, threshold: float, R: float) -> float:
    """Bénéfice net pour 1 000 employés, en unités de coût d'une intervention.

    On alerte les employés dont la probabilité dépasse ``threshold``. R = valeur
    espérée d'une intervention sur un futur partant / coût de l'intervention
    (R = taux de succès x coût d'un départ / coût d'une intervention). Chaque
    alerte coûte 1 ; chaque départ réellement ciblé rapporte R.
    """
    y, p = np.asarray(y), np.asarray(p)
    alert = p >= threshold
    tp = int((alert & (y == 1)).sum())
    return float((R * tp - alert.sum()) / len(y) * 1000)


def decision_table(y, p, ratios=(3, 5, 8.5, 12, 15, 20, 30, 50)) -> pd.DataFrame:
    """Seuil théorique optimal ``1/R`` (probabilités calibrées) et résultats sur le test.

    Un employé est ciblé si p x R > 1, c'est-à-dire p > 1/R : le gain espéré de l'alerte
    dépasse son coût. Comparé à « ne rien faire » (0) et « alerter tout le monde ».
    """
    y, p = np.asarray(y), np.asarray(p)
    rows = []
    for R in ratios:
        thr = 1.0 / R
        alert = p >= thr
        tp, n_alert = int((alert & (y == 1)).sum()), int(alert.sum())
        rows.append({
            "R": R, "seuil_optimal": thr, "alertes": n_alert, "départs_ciblés": tp,
            "précision": tp / n_alert if n_alert else np.nan, "rappel": tp / y.sum(),
            "bénéfice_modèle": net_benefit_per_1000(y, p, thr, R),
            "bénéfice_alerter_tous": net_benefit_per_1000(y, p, 0.0, R),
            "bénéfice_ne_rien_faire": 0.0,
        })
    out = pd.DataFrame(rows).set_index("R")
    out["gain_vs_sans_modèle"] = out["bénéfice_modèle"] - np.maximum(out["bénéfice_alerter_tous"], 0.0)
    return out
