# Inventaire des résultats — LOT 1

> Ce document distingue ce qui est **versionné** de ce qui est **généré puis ignoré** par Git, et donne les commandes exactes de régénération. Il ne modifie aucun résultat ni artefact.

---

## 1. Principe

- Les **sources** (CSV de résultats, figures, notebooks, code) sont versionnées.
- Les **artefacts générés** (modèles `joblib`, JSON de métriques, données traitées) sont recréés par les scripts et restent ignorés par `.gitignore`.
- Les fichiers `.gitkeep` de `models/`, `results/` et `data/processed/` sont versionnés pour conserver l'arborescence.

---

## 2. Ce qui est VERSIONNÉ

| Catégorie | Contenu | Règle Git |
|---|---|---|
| Jeu source | `data/raw/Employee_Attrition_DataSet.csv` | suivi |
| Résultats CSV | `results/*.csv` (22 fichiers) et `results/optimization/*.csv` (6 fichiers) | suivis |
| Synthèses clés | `results/model_comparison.csv`, `results/cv_summary.csv`, `results/tableau_metriques_rapport.csv`, `results/metrics_bootstrap_ic.csv`, `results/classement_multicritere.csv`, `results/explainability_top3.csv` | suivies |
| Statistiques | `results/stat_tests_vs_attrition.csv`, `results/stat_spearman_paires.csv`, `results/stat_normalite_shapiro.csv`, `results/stat_kruskal_wallis.csv`, `results/descriptive_stats.csv` | suivies |
| Figures du rapport | `reports/figures/*.png` (42 figures, numéros 01 à 42) | suivies |
| Figures du guide | `docs/figures/*.png` (3 figures) | suivies |
| Rapport scientifique | `docs/rapport/rapport.md`, `scripts/generate_rapport.py`, `docs/Rapport_MiniProject.pdf`, `docs/Rapport_Projet.pdf` | suivis |
| Notebooks | `notebooks/*.ipynb` (5 notebooks) | suivis |
| Marqueurs de dossiers | `models/.gitkeep`, `results/.gitkeep`, `data/processed/.gitkeep` | suivis |

Les deux tables qui prêtent à confusion sont bien versionnées et expliquées dans [`docs/METRIQUES.md`](../docs/METRIQUES.md) : elles correspondent à des seuils différents.

---

## 3. Ce qui est GÉNÉRÉ puis IGNORÉ

| Catégorie | Exemples | Producteur | Règle `.gitignore` |
|---|---|---|---|
| Modèles sérialisés | `models/*.joblib`, `models/cache/*.joblib` | `scripts/train_pipeline.py`, `scripts/train_models.py` | `models/*.joblib` |
| Métriques JSON | `results/metrics.json`, `results/roc_data.json`, `results/confusion_matrices.json`, `results/best_hyperparameters.json`, `results/cv_results_summary.json`, `results/feature_importances.json` | `scripts/train_pipeline.py` | `results/*.json` |
| Données traitées | `data/processed/*.csv`, dont `test_set.csv` | `scripts/run_preprocessing.py` | `data/processed/*` |
| Environnement et caches | `.venv/`, `__pycache__/`, `.pytest_cache/` | outils locaux | ignorés |

Aucun de ces artefacts ne doit être ajouté à un commit. Les régénérer localement avant d'exécuter les tests ou l'application.

---

## 4. Commandes exactes de régénération

Depuis la racine du dépôt, dans cet ordre :

```bash
python scripts/run_preprocessing.py
python scripts/train_pipeline.py
python scripts/train_models.py
python scripts/run_unsupervised.py
python scripts/run_unsupervised_advanced.py
python scripts/run_outliers.py
python scripts/run_explainability.py
python scripts/generate_rapport.py
python -m pytest tests -q
```

- `run_preprocessing.py` : CSV traités et `models/preprocessor.joblib` ; doit précéder les entraînements.
- `train_pipeline.py` : pipeline du guide, 6 modèles imposés, sélection sur F1 CV, JSON de métriques.
- `train_models.py` : analyse étendue, 14 configurations disponibles dans `src/models.py` (9 variantes évaluées), sélection sur AP CV, seuil optimal, bootstrap, `results/model_comparison.csv` et `results/cv_summary.csv`.
- `run_unsupervised.py` : PCA, KMeans, agglomératif, DBSCAN.
- `run_unsupervised_advanced.py` : choix de k, K-Medoids, DIANA, indices et figure 41.
- `run_outliers.py` : IQR, Z-score, Isolation Forest, LOF et figure 36.
- `run_explainability.py` : top-3 SHAP par employé, `results/explainability_top3.csv` et figure 42.
- `generate_rapport.py` : `docs/Rapport_MiniProject.pdf` et son alias `docs/Rapport_Projet.pdf` depuis `docs/rapport/rapport.md`.
- `pytest tests -q` : vérification des contrats ; **79 tests verts** attendus.

Sous Windows avec l'environnement du projet :

```powershell
.\.venv\Scripts\python.exe scripts\run_preprocessing.py
.\.venv\Scripts\python.exe scripts\train_pipeline.py
.\.venv\Scripts\python.exe scripts\train_models.py
.\.venv\Scripts\python.exe scripts\run_unsupervised.py
.\.venv\Scripts\python.exe scripts\run_unsupervised_advanced.py
.\.venv\Scripts\python.exe scripts\run_outliers.py
.\.venv\Scripts\python.exe scripts\run_explainability.py
.\.venv\Scripts\python.exe scripts\generate_rapport.py
.\.venv\Scripts\python.exe -m pytest tests -q
```

Script complémentaire, non requis pour les vérifications du lot : `python scripts/run_optimization.py` régénère `results/optimization/*.csv`.

---

## 5. Mapping fichier → producteur → consommateur

| Fichier | Producteur | Consommateur |
|---|---|---|
| `data/processed/*.csv` | `scripts/run_preprocessing.py` | `scripts/train_pipeline.py`, `scripts/train_models.py`, notebooks |
| `models/*.joblib` | `scripts/train_pipeline.py`, `scripts/train_models.py` | `app/utils.py`, pages de prédiction (interface hors périmètre de ce lot) |
| `results/metrics.json`, `results/roc_data.json`, `results/confusion_matrices.json`, `results/best_hyperparameters.json`, `results/cv_results_summary.json`, `results/feature_importances.json` | `scripts/train_pipeline.py` | `app/pages/3_Performance_Modeles.py` |
| `results/model_comparison.csv` | `scripts/train_models.py` | [`README.md`](../README.md) §6.2, [`docs/METRIQUES.md`](../docs/METRIQUES.md) |
| `results/cv_summary.csv` | `scripts/train_models.py` | [`docs/METRIQUES.md`](../docs/METRIQUES.md), rapport PDF |
| `results/tableau_metriques_rapport.csv` | `notebooks/05_metrics_analysis.ipynb` | `docs/Rapport_MiniProject.pdf`, README §6.2 |
| `docs/rapport/rapport.md` | lot 5 | `scripts/generate_rapport.py` |
| `docs/Rapport_MiniProject.pdf`, `docs/Rapport_Projet.pdf` | `scripts/generate_rapport.py` | évaluateur, README §9 |
| `results/stat_*.csv`, `results/descriptive_stats.csv` | `notebooks/01_eda_and_cleaning.ipynb`, `src/stats_tests.py` | README §5, rapport PDF |
| `results/optimization/*.csv` | `scripts/run_optimization.py` | rapport PDF, lot 5 du [`plan de projet`](../docs/PLAN_PROJET.md) |
| `results/explainability_top3.csv` | `scripts/run_explainability.py` | [`docs/GUIDE_IMPLEMENTATION_MODELE.md`](../docs/GUIDE_IMPLEMENTATION_MODELE.md), rapport PDF |
| `results/outliers_report.csv` | `scripts/run_outliers.py` | [`docs/OUTLIERS.md`](../docs/OUTLIERS.md), rapport PDF |
| `results/clustering_advanced.csv` | `scripts/run_unsupervised_advanced.py` | [`docs/CLUSTERING_AVANCE.md`](../docs/CLUSTERING_AVANCE.md), rapport PDF |
| `reports/figures/*.png` | `scripts/generate_figures.py`, notebooks | README, rapport PDF |
| `docs/figures/*.png` | `scripts/generate_figures.py` | rapport PDF |

Toute modification d'un producteur doit mettre à jour ce tableau et la documentation des métriques dans la même PR.
