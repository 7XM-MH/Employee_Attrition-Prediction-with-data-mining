"""
Pipeline d'entraînement complet pour la prédiction de l'attrition des employés.

Ce script entraîne 6 modèles avec recherche d'hyperparamètres et sauvegarde
tous les artefacts (modèles, métriques, courbes ROC, matrices de confusion,
importances des caractéristiques, etc.).

Usage:
    python scripts/train_pipeline.py
"""
import warnings
warnings.filterwarnings('ignore')

import json
import time
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import (
    train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)
from joblib import dump

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Employee_Attrition_DataSet.csv"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

NUMERICAL_FEATURES = [
    "Age", "JobLevel", "YearsAtCompany", "MonthlyIncome",
    "JobSatisfaction", "WorkLifeBalance", "DistanceFromHome",
    "PerformanceRating", "TrainingHoursLastYear",
]

CATEGORICAL_FEATURES = ["Department", "OverTime", "PromotionLast5Years"]

TARGET = "Attrition"
ID_COL = "EmployeeID"
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Helper – convert numpy types for JSON serialisation
# ---------------------------------------------------------------------------

def _make_serialisable(obj):
    """Recursively convert numpy types to native Python types."""
    if isinstance(obj, dict):
        return {k: _make_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serialisable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _save_json(data: dict, path: Path) -> None:
    """Save a dictionary to a JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_make_serialisable(data), f, indent=2, ensure_ascii=False)
    print(f"  ✔ Sauvegardé : {path.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("  PIPELINE D'ENTRAÎNEMENT — Attrition des Employés")
    print("=" * 70)

    # Ensure output directories exist
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load and prepare data
    # ------------------------------------------------------------------
    print("\n[1/6] Chargement des données …")
    df = pd.read_csv(DATA_PATH)
    print(f"  Lignes : {len(df)}  |  Colonnes : {df.shape[1]}")

    y = df[TARGET].map({"Yes": 1, "No": 0})
    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]

    print(f"  Distribution cible → No: {(y == 0).sum()}  |  Yes: {(y == 1).sum()}")

    # ------------------------------------------------------------------
    # 2. Train / test split
    # ------------------------------------------------------------------
    print("\n[2/6] Séparation train/test (80/20, stratifié) …")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"  Train : {len(X_train)}  |  Test : {len(X_test)}")

    # Save the test set with original columns + EmployeeID
    test_indices = X_test.index
    test_df = df.loc[test_indices].copy()
    test_df.to_csv(PROCESSED_DIR / "test_set.csv", index=False)
    print(f"  ✔ Jeu de test sauvegardé : data/processed/test_set.csv")

    # ------------------------------------------------------------------
    # 3. Preprocessor (ColumnTransformer)
    # ------------------------------------------------------------------
    print("\n[3/6] Construction du préprocesseur …")
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            (
                "cat",
                OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )
    print("  ✔ StandardScaler (numériques) + OneHotEncoder (catégorielles)")

    # ------------------------------------------------------------------
    # 4. Define models and hyperparameter grids
    # ------------------------------------------------------------------
    print("\n[4/6] Définition des modèles et grilles d'hyperparamètres …")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    model_configs = {
        "ZeroR": {
            "classifier": DummyClassifier(strategy="most_frequent"),
            "params": {},
            "search": "grid",
        },
        "KNN": {
            "classifier": KNeighborsClassifier(),
            "params": {
                "classifier__n_neighbors": [3, 5, 7, 9, 11],
                "classifier__weights": ["uniform", "distance"],
                "classifier__metric": ["euclidean", "manhattan"],
            },
            "search": "grid",
        },
        "DecisionTree": {
            "classifier": DecisionTreeClassifier(random_state=RANDOM_STATE),
            "params": {
                "classifier__max_depth": [3, 5, 7, 10, None],
                "classifier__min_samples_split": [2, 5, 10],
                "classifier__min_samples_leaf": [1, 2, 4],
                "classifier__criterion": ["gini", "entropy"],
            },
            "search": "grid",
        },
        "RandomForest": {
            "classifier": RandomForestClassifier(random_state=RANDOM_STATE),
            "params": {
                "classifier__n_estimators": [50, 100, 200],
                "classifier__max_depth": [5, 10, 15, None],
                "classifier__min_samples_split": [2, 5],
                "classifier__min_samples_leaf": [1, 2],
            },
            "search": "random",
            "n_iter": 20,
        },
        "NaiveBayes": {
            "classifier": GaussianNB(),
            "params": {
                "classifier__var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5],
            },
            "search": "grid",
        },
        "SVM": {
            "classifier": SVC(probability=True, random_state=RANDOM_STATE),
            "params": {
                "classifier__C": [0.1, 1, 10],
                "classifier__kernel": ["rbf", "linear"],
                "classifier__gamma": ["scale", "auto"],
            },
            "search": "random",
            "n_iter": 15,
        },
    }

    # ------------------------------------------------------------------
    # 5. Train models
    # ------------------------------------------------------------------
    print("\n[5/6] Entraînement des modèles …\n")

    all_metrics: dict = {}
    roc_data: dict = {}
    confusion_matrices: dict = {}
    feature_importances: dict = {}
    best_hyperparameters: dict = {}
    cv_results_summary: dict = {}
    best_f1 = -1.0
    best_model_name = ""

    for name, cfg in model_configs.items():
        print(f"  ─── {name} ───")
        t0 = time.time()

        # Build the pipeline
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", cfg["classifier"]),
        ])

        # Search
        if not cfg["params"]:
            # Zero-R: no search, just fit
            pipe.fit(X_train, y_train)
            search_obj = None
            best_hyperparameters[name] = {}
            cv_results_summary[name] = {"mean_cv_f1": None, "std_cv_f1": None}
        else:
            if cfg["search"] == "random":
                search_obj = RandomizedSearchCV(
                    pipe,
                    param_distributions=cfg["params"],
                    n_iter=cfg.get("n_iter", 10),
                    cv=cv,
                    scoring="f1",
                    refit=True,
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                    return_train_score=False,
                )
            else:
                search_obj = GridSearchCV(
                    pipe,
                    param_grid=cfg["params"],
                    cv=cv,
                    scoring="f1",
                    refit=True,
                    n_jobs=-1,
                    return_train_score=False,
                )

            search_obj.fit(X_train, y_train)
            pipe = search_obj.best_estimator_

            # Record best params (strip 'classifier__' prefix for readability)
            best_params_raw = search_obj.best_params_
            best_hyperparameters[name] = {
                k.replace("classifier__", ""): v for k, v in best_params_raw.items()
            }

            # CV results
            best_idx = search_obj.best_index_
            cv_results_summary[name] = {
                "mean_cv_f1": round(float(search_obj.cv_results_["mean_test_score"][best_idx]), 4),
                "std_cv_f1": round(float(search_obj.cv_results_["std_test_score"][best_idx]), 4),
            }

        # Evaluate on test set
        y_pred = pipe.predict(X_test)

        # Probabilities (for ROC)
        y_proba = None
        try:
            y_proba = pipe.predict_proba(X_test)[:, 1]
        except Exception:
            pass

        acc = round(accuracy_score(y_test, y_pred), 4)
        prec = round(precision_score(y_test, y_pred, zero_division=0), 4)
        rec = round(recall_score(y_test, y_pred, zero_division=0), 4)
        f1 = round(f1_score(y_test, y_pred, zero_division=0), 4)
        auc = round(roc_auc_score(y_test, y_proba), 4) if y_proba is not None else None

        all_metrics[name] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": auc,
        }

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        confusion_matrices[name] = cm.tolist()

        # ROC data
        if y_proba is not None:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_data[name] = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "auc": auc,
            }

        # Feature importances (tree-based models only)
        if name in ("DecisionTree", "RandomForest"):
            clf = pipe.named_steps["classifier"]
            importances = clf.feature_importances_
            feature_names = pipe.named_steps["preprocessor"].get_feature_names_out()
            feat_imp = dict(zip(feature_names.tolist(), importances.tolist()))
            # Sort descending
            feat_imp = dict(sorted(feat_imp.items(), key=lambda x: x[1], reverse=True))
            feature_importances[name] = feat_imp

        # Save the pipeline
        dump(pipe, MODELS_DIR / f"{name}_pipeline.joblib")

        # Track best model by F1
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name

        elapsed = time.time() - t0
        roc_str = f"{auc}" if auc is not None else "N/A"
        print(f"    Accuracy={acc}  Precision={prec}  Recall={rec}  F1={f1}  ROC-AUC={roc_str}")
        if name in best_hyperparameters and best_hyperparameters[name]:
            print(f"    Meilleurs hyperparamètres : {best_hyperparameters[name]}")
        if name in cv_results_summary and cv_results_summary[name]["mean_cv_f1"] is not None:
            print(f"    CV F1 : {cv_results_summary[name]['mean_cv_f1']} ± {cv_results_summary[name]['std_cv_f1']}")
        print(f"    Temps : {elapsed:.1f}s\n")

    # ------------------------------------------------------------------
    # 6. Save all artifacts
    # ------------------------------------------------------------------
    print("[6/6] Sauvegarde des artefacts …\n")

    # Best model — copy the already-saved pipeline as best_model.joblib
    print(f"  ★ Meilleur modèle (F1 test) : {best_model_name} (F1 = {best_f1})")
    from joblib import load as jl_load
    best_pipe = jl_load(MODELS_DIR / f"{best_model_name}_pipeline.joblib")
    dump(best_pipe, MODELS_DIR / "best_model.joblib")
    print(f"  ✔ Sauvegardé : models/best_model.joblib")

    _save_json(all_metrics, RESULTS_DIR / "metrics.json")
    _save_json(roc_data, RESULTS_DIR / "roc_data.json")
    _save_json(confusion_matrices, RESULTS_DIR / "confusion_matrices.json")
    _save_json(feature_importances, RESULTS_DIR / "feature_importances.json")
    _save_json(best_hyperparameters, RESULTS_DIR / "best_hyperparameters.json")
    _save_json(cv_results_summary, RESULTS_DIR / "cv_results_summary.json")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  RÉSUMÉ DES RÉSULTATS")
    print("=" * 70)
    header = f"  {'Modèle':<16} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>7} {'ROC-AUC':>9}"
    print(header)
    print("  " + "-" * 62)
    for name, m in all_metrics.items():
        roc_str = f"{m['roc_auc']}" if m["roc_auc"] is not None else "N/A"
        marker = " ★" if name == best_model_name else ""
        print(f"  {name:<16} {m['accuracy']:>9} {m['precision']:>10} {m['recall']:>8} {m['f1']:>7} {roc_str:>9}{marker}")

    print("\n  ✔ Pipeline terminé avec succès !")
    print("=" * 70)


if __name__ == "__main__":
    main()
