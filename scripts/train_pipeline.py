"""
Pipeline d'entraînement complet pour la prédiction de l'attrition des employés.

Point d'entrée unique (voir docs/GUIDE_IMPLEMENTATION_MODELE.md) :
    python scripts/train_pipeline.py

Le script entraîne les six modèles du cours (Zero-R, K-NN, arbre de décision,
forêt aléatoire, Naive Bayes, SVM) et sauvegarde :

    models/<NomModele>_pipeline.joblib   (un seul Pipeline : préprocesseur + classifieur)
    models/best_model.joblib             (bundle {model, threshold} du meilleur modèle CV)
    results/metrics.json                 (métriques de TEST, evaluation finale)
    results/roc_data.json
    results/confusion_matrices.json      (ordre des classes : [0, 1] = [No, Yes])
    results/best_hyperparameters.json
    results/cv_results_summary.json
    results/feature_importances.json
    results/selection_metadata.json      (nom/critère du meilleur modèle, ordre des classes)
    data/processed/test_set.csv

Points de contrat :
- le schéma des 12 features vient de src/preprocessing.py (aucun troisième schéma) ;
- le split stratifié précède tout ajustement et le test n'est jamais réutilisé
  pour choisir le gagnant ; la sélection se fait sur le F1 moyen de la CV ;
- la classe positive est `Yes` -> 1 ; `predict_proba` est lu à l'indice de la classe 1 ;
- SMOTE n'est pas appliqué : l'enseignant ne l'a pas demandé. Si requis, il devra
  être inséré dans chaque pli via imblearn.pipeline.Pipeline, jamais avant le split.
"""
import sys
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Console Windows (cp1252) : éviter UnicodeEncodeError sur les messages JSON/UI.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")

import json
import time

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.models import load_bundle, save_bundle
from src.preprocessing import (
    BINARY_COLS,
    FEATURE_COLS,
    ID_COL,
    NOMINAL_COLS,
    NUMERIC_COLS,
    ORDINAL_COLS,
    TARGET,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Le schéma a été renommé dans src.preprocessing (lot 2) : les anciens alias
# ALL_FEATURES / DATA_PATH / prepare_data n'existent plus. On garde le contrat
# historique de CE script uniquement, sans modifier le module partagé.
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Employee_Attrition_DataSet.csv"
ALL_FEATURES = FEATURE_COLS
NUMERICAL_FEATURES = NUMERIC_COLS + ORDINAL_COLS
CATEGORICAL_FEATURES = NOMINAL_COLS + BINARY_COLS

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SPLITS = 5
POSITIVE_LABEL = 1  # Yes, d'après src.preprocessing.prepare_data
TRAIN_COMMAND = "python scripts/train_pipeline.py"

# Mapping de la cible : identique à src.preprocessing.prepare_data (pas de 2e schéma).
TARGET_VALUES = {"No": 0, "Yes": 1}
CLASS_LABELS = ["No", "Yes"]
CLASS_ORDER = [TARGET_VALUES[label] for label in CLASS_LABELS]


def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Retire EmployeeID de X et encode la cible No->0 / Yes->1.

    L'encodage est fait ici, avant le split, car il est déterministe et
    n'apprend aucune statistique : il n'y a donc aucun risque de fuite.
    """
    X = df[ALL_FEATURES].copy()
    y = df[TARGET].map(TARGET_VALUES).astype(int)
    return X, y


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_serialisable(obj):
    """Convertit récursivement les types NumPy en types Python natifs (JSON)."""
    if isinstance(obj, dict):
        return {str(k): _make_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serialisable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _save_json(data: dict, path: Path) -> None:
    """Sauvegarde un dictionnaire en JSON (indenté, UTF-8, types Python natifs)."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_make_serialisable(data), f, indent=2, ensure_ascii=False)
    print(f"  ✔ Sauvegardé : {path.relative_to(PROJECT_ROOT)}")


def validate_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Vérifie le contrat de données avant tout split.

    Lève ValueError avec un message explicite si le schéma est invalide.
    Le CSV est la source de vérité (docs/GUIDE_IMPLEMENTATION_MODELE.md §6).
    """
    required = set(ALL_FEATURES) | {TARGET}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            "Schéma invalide : colonnes attendues absentes du CSV : " + ", ".join(missing)
        )

    if df[TARGET].isna().any():
        raise ValueError(f"Schéma invalide : la cible {TARGET!r} contient des valeurs manquantes.")

    unexpected = sorted(set(df[TARGET].unique()) - set(TARGET_VALUES))
    if unexpected:
        raise ValueError(
            f"Schéma invalide : {TARGET!r} doit contenir uniquement No/Yes, "
            f"valeurs inattendues : {unexpected}"
        )

    for label in CLASS_LABELS:
        if (df[TARGET] == label).sum() == 0:
            raise ValueError(f"Schéma invalide : aucune ligne de classe {label!r} dans {TARGET!r}.")

    if ID_COL in df.columns:
        ids = df[ID_COL].astype(str)
        if not ids.is_unique:
            raise ValueError(f"Schéma invalide : {ID_COL!r} contient des identifiants dupliqués.")
        print(f"  ✔ {ID_COL} : {len(ids)} identifiants uniques (exclu de X)")

    # Les features numériques doivent être convertibles en nombres.
    df = df.copy()
    for col in NUMERICAL_FEATURES:
        try:
            converted = pd.to_numeric(df[col], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Schéma invalide : {col!r} n'est pas numérique : {exc}") from exc
        if converted.isna().any():
            raise ValueError(f"Schéma invalide : {col!r} contient des valeurs manquantes.")
        df[col] = converted

    return df


def build_preprocessor() -> ColumnTransformer:
    """ColumnTransformer ajusté uniquement à l'intérieur du Pipeline/CV."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def build_model_configs() -> dict:
    """Les six modèles classiques et leurs grilles de recherche."""
    return {
        "ZeroR": {
            "classifier": DummyClassifier(strategy="most_frequent"),
            "params": {},
            "search": None,
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
            "n_iter": 10,
        },
        "NaiveBayes": {
            "classifier": GaussianNB(),
            "params": {"classifier__var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]},
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
            "n_iter": 10,
        },
    }


def positive_class_index(pipe: Pipeline) -> int:
    """Index de la classe positive (`Yes` -> 1) dans predict_proba.

    Un Pipeline n'expose pas toujours `classes_` : on interroge le classifieur
    final et on vérifie l'ordre réel plutôt que de supposer l'indice 1.
    """
    classifier = pipe.named_steps.get("classifier", pipe)
    classes = list(getattr(classifier, "classes_", CLASS_ORDER))
    if POSITIVE_LABEL in classes:
        return classes.index(POSITIVE_LABEL)
    return CLASS_ORDER.index(POSITIVE_LABEL)


def compute_feature_importances(
    name: str, pipe: Pipeline, X_train: pd.DataFrame, y_train: pd.Series
) -> dict:
    """Importances natives (arbres) ou permutation sur un échantillon du train.

    Zero-R est constant : aucune importance n'est disponible, il est donc omis
    de `feature_importances.json` (page Facteurs : modèles disponibles seulement).
    """
    if name == "ZeroR":
        return {}
    classifier = pipe.named_steps["classifier"]
    if hasattr(classifier, "feature_importances_"):
        names = pipe.named_steps["preprocessor"].get_feature_names_out()
        values = classifier.feature_importances_
        ranked = sorted(zip(names, values), key=lambda item: item[1], reverse=True)
        return {str(feature): float(value) for feature, value in ranked}

    sample_size = min(500, len(X_train))
    X_sample = X_train.sample(n=sample_size, random_state=RANDOM_STATE)
    y_sample = y_train.loc[X_sample.index]
    result = permutation_importance(
        pipe,
        X_sample,
        y_sample,
        scoring="f1",
        n_repeats=3,
        random_state=RANDOM_STATE,
        n_jobs=1,  # n_jobs=1 pour compatibilité Windows (évite TerminatedWorkerError)
    )
    ranked = sorted(
        zip(ALL_FEATURES, result.importances_mean), key=lambda item: item[1], reverse=True
    )
    return {str(feature): float(value) for feature, value in ranked}


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 70)
    print("  PIPELINE D'ENTRAÎNEMENT — Attrition des Employés")
    print("=" * 70)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Chargement et validation du schéma ---------------------------------
    print("\n[1/6] Chargement et validation des données …")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"CSV introuvable : {DATA_PATH}")
    df = validate_schema(pd.read_csv(DATA_PATH))
    X, y = prepare_data(df)
    print(f"  Lignes : {len(df)}  |  Colonnes : {df.shape[1]}  |  Features : {len(ALL_FEATURES)}")
    print(
        f"  Distribution cible → No: {(y == TARGET_VALUES['No']).sum()}  |  "
        f"Yes: {(y == TARGET_VALUES['Yes']).sum()}"
    )

    # 2. Split stratifié AVANT tout ajustement ------------------------------
    print("\n[2/6] Séparation train/test (80/20, stratifié) …")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"  Train : {len(X_train)}  |  Test : {len(X_test)} (réservé à l'évaluation finale)")
    test_df = df.loc[X_test.index].copy()
    test_path = PROCESSED_DIR / "test_set.csv"
    test_df.to_csv(test_path, index=False)
    print(f"  ✔ Jeu de test sauvegardé : {test_path.relative_to(PROJECT_ROOT)}")

    # 3. Validation croisée --------------------------------------------------
    print("\n[3/6] Validation croisée …")
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    print(f"  StratifiedKFold(n_splits={N_SPLITS}, shuffle=True, random_state={RANDOM_STATE})")

    # 4. Modèles -------------------------------------------------------------
    print("\n[4/6] Définition des modèles et grilles d'hyperparamètres …")
    model_configs = build_model_configs()
    print("  " + ", ".join(model_configs))

    # 5. Entraînement / sélection sur CV / évaluation finale sur test --------
    print("\n[5/6] Entraînement des modèles …\n")
    test_metrics: dict = {}
    roc_data: dict = {}
    confusion_matrices: dict = {}
    feature_importances: dict = {}
    best_hyperparameters: dict = {}
    cv_results_summary: dict = {}

    best_model_name = ""
    best_cv_f1 = -1.0
    best_cv_std = 0.0
    best_pipe = None

    for name, cfg in model_configs.items():
        print(f"  ─── {name} ───")
        t0 = time.time()
        pipe = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", cfg["classifier"]),
        ])

        if cfg["search"] is None:
            # Zero-R : pas d'hyperparamètre, mais un score CV réel (pas un score nul).
            cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1", n_jobs=1)  # n_jobs=1 Windows compat
            mean_cv_f1 = float(cv_scores.mean())
            std_cv_f1 = float(cv_scores.std())
            pipe.fit(X_train, y_train)
            search_obj = None
            best_hyperparameters[name] = {}
        else:
            search_kwargs = dict(
                estimator=pipe,
                cv=cv,
                scoring="f1",  # F1 de la classe positive Yes (encodée 1)
                refit=True,
                n_jobs=1,  # n_jobs=1 pour compatibilité Windows (évite TerminatedWorkerError)
                return_train_score=False,
            )
            if cfg["search"] == "random":
                search_kwargs["random_state"] = RANDOM_STATE
                search_obj = RandomizedSearchCV(
                    param_distributions=cfg["params"],
                    n_iter=cfg.get("n_iter", 10),
                    **search_kwargs,
                )
            else:
                search_obj = GridSearchCV(param_grid=cfg["params"], **search_kwargs)

            search_obj.fit(X_train, y_train)
            pipe = search_obj.best_estimator_
            best_idx = search_obj.best_index_
            mean_cv_f1 = float(search_obj.cv_results_["mean_test_score"][best_idx])
            std_cv_f1 = float(search_obj.cv_results_["std_test_score"][best_idx])
            best_hyperparameters[name] = {
                str(k).replace("classifier__", ""): v
                for k, v in search_obj.best_params_.items()
            }

        cv_results_summary[name] = {
            "mean_cv_f1": round(mean_cv_f1, 4),
            "std_cv_f1": round(std_cv_f1, 4),
        }

        # Évaluation finale sur le test (jamais utilisée pour choisir le gagnant).
        y_pred = pipe.predict(X_test)
        yes_idx = positive_class_index(pipe)
        y_proba = pipe.predict_proba(X_test)[:, yes_idx]

        acc = round(accuracy_score(y_test, y_pred), 4)
        prec = round(precision_score(y_test, y_pred, zero_division=0), 4)
        rec = round(recall_score(y_test, y_pred, zero_division=0), 4)
        f1 = round(f1_score(y_test, y_pred, zero_division=0), 4)
        auc = round(roc_auc_score(y_test, y_proba), 4)

        test_metrics[name] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": auc,
        }
        confusion_matrices[name] = confusion_matrix(y_test, y_pred).tolist()
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_data[name] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": auc}
        feature_importances[name] = compute_feature_importances(name, pipe, X_train, y_train)

        pipe_path = MODELS_DIR / f"{name}_pipeline.joblib"
        dump(pipe, pipe_path)

        # Sélection uniquement sur la CV (le test reste intact pour le choix).
        if mean_cv_f1 > best_cv_f1:
            best_cv_f1 = mean_cv_f1
            best_cv_std = std_cv_f1
            best_model_name = name
            best_pipe = pipe

        elapsed = time.time() - t0
        print(f"    CV F1      : {mean_cv_f1:.4f} ± {std_cv_f1:.4f}")
        print(f"    TEST       : accuracy={acc}  precision={prec}  recall={rec}  f1={f1}  roc_auc={auc}")
        print(f"    Temps      : {elapsed:.1f}s\n")

    # 6. Meilleur modèle et artefacts ----------------------------------------
    print("[6/6] Sauvegarde des artefacts …\n")
    # Le fichier applicatif est un bundle {model, threshold}, pas un Pipeline nu :
    # c'est le contrat consommé par src.models.load_bundle et par l'interface.
    best_path = save_bundle(
        best_model_name,
        best_pipe,
        0.5,  # aucun optimisation de seuil dans ce pipeline : décision à 0,5
        {"selection_criterion": "mean_cv_f1", "selection_score": best_cv_f1},
    )
    # Vérification immédiate : le fichier doit se charger et prédire une probabilité finie.
    reloaded = load_bundle(best_path)
    check_proba = reloaded["model"].predict_proba(X_test.head(1))[0, positive_class_index(reloaded["model"])]
    if not np.isfinite(check_proba):
        raise RuntimeError("predict_proba du meilleur modèle renvoie une valeur non finie.")
    print(f"  ★ Meilleur modèle (sélection CV F1) : {best_model_name} "
          f"(CV F1 = {best_cv_f1:.4f} ± {best_cv_std:.4f})")
    print(f"  ✔ Sauvegardé : {best_path.relative_to(PROJECT_ROOT)}")

    _save_json(test_metrics, RESULTS_DIR / "metrics.json")
    _save_json(roc_data, RESULTS_DIR / "roc_data.json")
    _save_json(confusion_matrices, RESULTS_DIR / "confusion_matrices.json")
    _save_json(best_hyperparameters, RESULTS_DIR / "best_hyperparameters.json")
    _save_json(cv_results_summary, RESULTS_DIR / "cv_results_summary.json")
    _save_json(feature_importances, RESULTS_DIR / "feature_importances.json")

    # Métadonnées explicites : critère de sélection, ordre des classes, producteur.
    # Consommateurs : app/Home.py (nom/critière du meilleur modèle) et
    # app/pages/3_Performance_Modeles.py (séparation CV / test).
    selection_metadata = {
        "best_model": best_model_name,
        "selection_criterion": "mean_cv_f1",
        "selection_score": round(best_cv_f1, 4),
        "selection_score_std": round(best_cv_std, 4),
        "selection_rule": (
            "Le modèle ayant le F1 moyen maximal sur la validation croisée du jeu "
            "d'apprentissage est retenu ; le jeu de test ne sert qu'à l'évaluation finale."
        ),
        "cv": {
            "splitter": "StratifiedKFold",
            "n_splits": N_SPLITS,
            "shuffle": True,
            "random_state": RANDOM_STATE,
            "scoring": "f1",
        },
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "positive_class": {"label": "Yes", "encoded_value": POSITIVE_LABEL},
        "class_order": CLASS_ORDER,
        "class_labels": CLASS_LABELS,
        "features": ALL_FEATURES,
        "train_command": TRAIN_COMMAND,
    }
    _save_json(selection_metadata, RESULTS_DIR / "selection_metadata.json")

    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  RÉSUMÉ — sélection CV et évaluation TEST distinctes")
    print("=" * 70)
    header = (
        f"  {'Modèle':<16} {'CV F1':>9} {'TEST F1':>9} {'TEST AUC':>9}"
    )
    print(header)
    print("  " + "-" * 48)
    for name in model_configs:
        cv_f1 = cv_results_summary[name]["mean_cv_f1"]
        marker = " ★ sélectionné (CV)" if name == best_model_name else ""
        m = test_metrics[name]
        print(f"  {name:<16} {cv_f1:>9.4f} {m['f1']:>9.4f} {m['roc_auc']:>9.4f}{marker}")

    print(f"\n  Meilleur modèle : {best_model_name} — critère : mean_cv_f1")
    print(f"  ✔ Pipeline terminé avec succès !")
    print("=" * 70)


if __name__ == "__main__":
    main()
