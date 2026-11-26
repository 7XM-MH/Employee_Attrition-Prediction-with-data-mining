# Employee Attrition — Mini-projet Data Mining 2026/2027

> Classification binaire supervisée + clustering non supervisé — pipeline end-to-end.
> Jeu de données : 10 000 employés, 12 variables explicatives, cible `Attrition` (`No` / `Yes`).

> [!WARNING]
> Projet **académique et fictif**. Les données sont synthétiques. Aucune prédiction ne doit être utilisée pour une décision RH réelle.

> Pilotage : [plan de projet](docs/PLAN_PROJET.md) · [convention de contribution](CONTRIBUTING.md) · [métriques](docs/METRIQUES.md) · [inventaire des résultats](results/README_resultats.md).

---

## 1. Compréhension métier

Le renouvellement du personnel coûte cher (recrutement, intégration, perte de compétences, baisse de moral). L'objectif est d'anticiper les départs en identifiant les facteurs individuels et organisationnels associés à l'attrition.

**Question métier :** les caractéristiques démographiques, hiérarchiques et d'environnement de travail (revenu, ancienneté, heures supplémentaires, satisfaction, équilibre vie pro/perso…) permettent-elles de prédire si un employé va quitter l'entreprise ?

**Tâche Data Mining retenue :** classification binaire supervisée (`No → 0`, `Yes → 1`), complétée par une analyse non supervisée descriptive. Aucun deep learning ni LLM n'est utilisé comme modèle prédictif.

---

## 2. Jeu de données

| Propriété | Valeur |
|---|---|
| Fichier | `data/raw/Employee_Attrition_DataSet.csv` |
| Dimensions | **10 000 lignes × 14 colonnes** |
| Identifiant | `EmployeeID` — exclu de `X` |
| Cible | `Attrition` — `No` : 9 097 (90,97 %), `Yes` : 903 (9,03 %) |
| Variables explicatives | **12** : 5 quantitatives, 4 ordinales, 2 binaires, 1 nominale |
| Valeurs manquantes | 0 |
| Doublons | 0 |
| Nature | Dataset synthétique à usage pédagogique |

Les 12 features sont : `Age`, `YearsAtCompany`, `MonthlyIncome`, `DistanceFromHome`, `TrainingHoursLastYear`, `JobLevel`, `JobSatisfaction`, `WorkLifeBalance`, `PerformanceRating`, `OverTime`, `PromotionLast5Years`, `Department`.

Références détaillées : [`DISCOVERY.md`](DISCOVERY.md) et [`docs/data_dictionary.md`](docs/data_dictionary.md) (types, échelles de mesure, plages, statistiques descriptives).

> Le déséquilibre d'environ 10:1 justifie le **F1-score** et l'**Average Precision** comme critères principaux, plutôt que l'accuracy (un modèle qui prédit toujours `No` obtient déjà 0,9095 d'accuracy).

---

## 3. Démarche CRISP-DM (6 étapes)

| Étape | Livrables concrets |
|---|---|
| **1. Compréhension métier** | §1 ci-dessus, [`docs/GUIDE_IMPLEMENTATION_MODELE.md`](docs/GUIDE_IMPLEMENTATION_MODELE.md) |
| **2. Compréhension des données** | [`DISCOVERY.md`](DISCOVERY.md), [`notebooks/01_eda_and_cleaning.ipynb`](notebooks/01_eda_and_cleaning.ipynb), [`results/descriptive_stats.csv`](results/descriptive_stats.csv), figures `01` à `08` |
| **3. Préparation des données** | [`src/preprocessing.py`](src/preprocessing.py), [`scripts/run_preprocessing.py`](scripts/run_preprocessing.py), [`docs/OUTLIERS.md`](docs/OUTLIERS.md), [`notebooks/02_preprocessing_encoding.ipynb`](notebooks/02_preprocessing_encoding.ipynb) |
| **4. Modélisation** | [`scripts/train_pipeline.py`](scripts/train_pipeline.py) (pipeline du guide), [`scripts/train_models.py`](scripts/train_models.py) + [`src/models.py`](src/models.py) (analyse étendue), [`results/cv_summary.csv`](results/cv_summary.csv), figures `09` à `16` |
| **5. Évaluation** | [`results/model_comparison.csv`](results/model_comparison.csv), `results/*.json`, [`tests/`](tests/), figures `17` à `23` |
| **6. Déploiement et surveillance** | Modèles `joblib` dans `models/`, application Streamlit dans `app/` ; surveillance et concept drift discutés en limite (§9) |

Le cycle est itératif : les résultats d'évaluation (étape 5) conduisent à ajuster la préparation (étape 3) et la modélisation (étape 4).

L'interface Streamlit est livrée par son propriétaire (lot 6) : `streamlit run app/Home.py`. Elle charge le bundle `models/best_model.joblib` via `src.models.load_bundle` (probabilité de la classe `Yes` et seuil de décision du bundle ; 0,5 pour l'artefact historique du guide), utilise les 12 variables de `src.preprocessing.FEATURE_COLS` (`No→0` / `Yes→1`, `EmployeeID` et `Attrition` exclus de `X`) et délègue l'explicabilité à `src.explainability`. Les pages de prédiction individuelle, de prédiction en lot et d'analyse SHAP se trouvent dans [`app/pages/`](app/pages/).

---

## 4. Notebooks et scripts

| Fichier | Rôle |
|---|---|
| `notebooks/01_eda_and_cleaning.ipynb` | EDA, audit qualité, détection/écrêtage IQR des outliers, distributions, corrélations, χ², Mann-Whitney U, Spearman, correction Benjamini-Hochberg |
| `notebooks/02_preprocessing_encoding.ipynb` | Encodage Y/X, normalisation, split train/test sans fuite |
| `notebooks/03_model_training_evaluation.ipynb` | Réglage des modèles, seuil optimal, calibration, importance par permutation |
| `notebooks/04_optimization.ipynb` | Preuves d'optimisation, CV imbriquée, analyse coût-bénéfice du seuil |
| `notebooks/05_metrics_analysis.ipynb` | Tableau multicritère, matrices de confusion, ROC/PR, intervalles de confiance |
| `scripts/run_preprocessing.py` | Prétraitement reproductible brut → `data/processed/` + `models/preprocessor.joblib` |
| `scripts/train_pipeline.py` | Pipeline du guide : 6 modèles imposés, StratifiedKFold(5), artefacts JSON |
| `scripts/train_models.py` | Analyse étendue : 14 configurations définies / 9 variantes évaluées, recherche AP, bootstrap, optimisation du seuil |
| `scripts/run_unsupervised.py` | KMeans, PCA, agglomératif, DBSCAN |
| [`scripts/run_unsupervised_advanced.py`](scripts/run_unsupervised_advanced.py) | Choix de k, K-Medoids, DIANA, cosinus, indices de validation → `results/clustering_advanced.csv` et figures 37 à 41 |
| [`scripts/run_outliers.py`](scripts/run_outliers.py) | Détection des outliers : IQR, Z-score, Isolation Forest, LOF → `results/outliers_report.csv` et figure 36 |
| [`scripts/run_explainability.py`](scripts/run_explainability.py) | Explicabilité locale SHAP (top 3 par employé) → `results/explainability_top3.csv` et figure 42 |
| [`scripts/generate_rapport.py`](scripts/generate_rapport.py) | Rapport PDF depuis [`docs/rapport/rapport.md`](docs/rapport/rapport.md) → `docs/Rapport_MiniProject.pdf` et `docs/Rapport_Projet.pdf` |
| `scripts/generate_figures.py`, `scripts/generate_notebooks.py` | Génération des livrables graphiques / notebooks |

Analyse des outliers détaillée dans [`docs/OUTLIERS.md`](docs/OUTLIERS.md), avec la comparaison graphique [`reports/figures/36_outliers_methodes.png`](reports/figures/36_outliers_methodes.png).

---

## 5. Tests statistiques (α = 0,05, correction BH)

Extrait de [`results/stat_tests_vs_attrition.csv`](results/stat_tests_vs_attrition.csv) — 12 variables testées, 9 significatives après correction :

| Variable | Test | p-value ajustée (BH) | Taille d'effet | Sens |
|---|---|---:|---:|---|
| `JobSatisfaction` | Mann-Whitney U | 3,2 × 10⁻¹⁶ | \|r\| = 0,164 (faible) | plus basse chez les partants |
| `WorkLifeBalance` | Mann-Whitney U | 1,1 × 10⁻¹² | \|r\| = 0,140 (faible) | plus bas chez les partants |
| `JobLevel` | Mann-Whitney U | 8,0 × 10⁻¹⁰ | \|r\| = 0,123 (faible) | plus bas chez les partants |
| `OverTime` | χ² | 2,1 × 10⁻¹¹ | V = 0,069 (négligeable) | lien significatif |
| `Department` | χ² | 0,583 | V = 0,023 | non significatif |

Fichiers complémentaires : [`stat_spearman_paires.csv`](results/stat_spearman_paires.csv), [`stat_normalite_shapiro.csv`](results/stat_normalite_shapiro.csv), [`stat_kruskal_wallis.csv`](results/stat_kruskal_wallis.csv).

---

## 6. Résultats supervisés

### 6.1 Pipeline du guide — 6 modèles imposés

`python scripts/train_pipeline.py` — split 80/20 stratifié, `StratifiedKFold(n_splits=5)`, `GridSearchCV`/`RandomizedSearchCV`, sélection sur le **F1 moyen en validation croisée**, test réservé à l'évaluation finale.

| Modèle | F1 CV | Accuracy | Precision | Recall | F1 test | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| ZeroR | 0,0000 | 0,9095 | 0,0000 | 0,0000 | 0,0000 | 0,5000 |
| KNN | 0,0542 | 0,8930 | 0,1633 | 0,0442 | 0,0696 | 0,5424 |
| **DecisionTree** | **0,1365** | 0,8645 | 0,1484 | 0,1050 | **0,1230** | 0,5406 |
| RandomForest | 0,0055 | 0,9090 | 0,0000 | 0,0000 | 0,0000 | 0,5921 |
| NaiveBayes | 0,0833 | 0,8915 | 0,2000 | 0,0663 | 0,0996 | 0,6206 |
| SVM | 0,0027 | 0,9085 | 0,0000 | 0,0000 | 0,0000 | 0,5404 |

**Modèle retenu : `DecisionTree`** — F1 CV = 0,1365 ± 0,0230, F1 test = 0,1230, ROC-AUC = 0,5406. Son excédent par rapport à ZeroR reste faible : le signal prédictif de ce jeu synthétique est limité.

### 6.2 Analyse étendue — 14 configurations définies, 9 variantes évaluées et optimisation du seuil

`python scripts/train_models.py` — sélection sur l'**Average Precision** en CV 5 plis, intervalles de confiance bootstrap, seuil de décision optimisé, calibration, gain/lift, importance par permutation.

| Modèle | AP CV | ROC-AUC test | PR-AUC test | Seuil optimal | F1 au seuil |
|---|---:|---:|---:|---:|---:|
| Zero-R | — | 0,5000 | 0,0905 | 0,500 | 0,0000 |
| K-NN | 0,1568 | 0,6511 | 0,1691 | 0,134 | 0,2365 |
| Arbre de décision | 0,1456 | 0,6083 | 0,1369 | 0,121 | 0,1972 |
| Random Forest | 0,1691 | 0,6550 | 0,1630 | 0,111 | 0,2351 |
| Naive Bayes (Gaussian) | 0,1424 | 0,6254 | 0,1435 | 0,096 | 0,2056 |
| Naive Bayes (Bernoulli) | 0,1615 | 0,6341 | 0,1520 | 0,105 | 0,2194 |
| SVM | 0,1499 | 0,6554 | 0,1576 | 0,110 | 0,2391 |
| Régression logistique *(extension)* | 0,1525 | 0,6564 | 0,1595 | 0,116 | 0,2392 |
| **Gradient Boosting** *(extension)* | **0,1729** | **0,6648** | **0,1732** | 0,118 | **0,2527** |

**Point clé :** au seuil par défaut de 0,5, **tous** les modèles ont un F1 de 0 (aucun employé prédit partant). L'optimisation du seuil sur les données d'entraînement/validation est indispensable avec 9 % de classe positive. Le choix du modèle reste fait sur la CV, jamais sur le test.

Les deux pipelines ont des critères de sélection différents : `DecisionTree` est le modèle du **guide** (F1 CV), `Gradient Boosting` est le meilleur de l'**analyse étendue** (AP CV). Les deux résultats sont conservés et documentés séparément.

Figures : [`reports/figures/17_comparaison_cv.png`](reports/figures/17_comparaison_cv.png), [`18_roc_pr_test.png`](reports/figures/18_roc_pr_test.png), [`19_confusions_seuil_optimise.png`](reports/figures/19_confusions_seuil_optimise.png), [`20_calibration.png`](reports/figures/20_calibration.png), [`21_gain_lift.png`](reports/figures/21_gain_lift.png), [`22_choix_du_seuil.png`](reports/figures/22_choix_du_seuil.png), [`23_importance_permutation.png`](reports/figures/23_importance_permutation.png).

---

## 7. Clustering non supervisé

`python scripts/run_unsupervised.py` — `Attrition` n'est **jamais** utilisée pendant l'ajustement ; elle sert uniquement à interpréter les groupes après coup.

| Méthode | Paramètres | Résultat | Silhouette |
|---|---|---|---:|
| PCA | 2 composantes | 40,58 % de variance expliquée | — |
| KMeans | k = 4 | 4 groupes (1 284 / 2 074 / 3 284 / 3 358) | 0,1051 |
| AgglomerativeClustering | 4 groupes, connectivité 10-PPV | 4 groupes (4 245 / 3 118 / 1 786 / 851) | 0,0732 |
| DBSCAN | eps = 0,18 ; min_samples = 25 | 2 groupes + 1 394 points de bruit | 0,2675 |

Le taux de départ du cluster KMeans le plus à risque est de 10,9 % contre 5,8 % pour le plus faible : une Segmentation exploitable reste limitée (silhouettes faibles).

---

## 8. Reproduction

```bash
python -m venv venv
venv\Scripts\activate            # Linux/macOS : source venv/bin/activate
pip install -r requirements.txt

python scripts/run_preprocessing.py   # données traitées + preprocessor.joblib
python scripts/train_pipeline.py      # pipeline du guide (6 modèles)
python scripts/train_models.py        # analyse étendue (variantes évaluées, plus long)
python scripts/run_unsupervised.py    # clustering (4 modèles historiques)
python scripts/run_unsupervised_advanced.py  # clustering avancé (lot 3)
python scripts/run_outliers.py        # détection avancée des outliers
pytest tests -q                       # suite de tests
streamlit run app/Home.py             # application web
```

Les `.joblib` et `results/*.json` sont des artefacts générés, **ignorés par Git** (voir `.gitignore`) : ils sont recréés par les scripts ci-dessus. Les `.csv` de résultats et les figures de `reports/figures/` sont versionnés.

> [!NOTE]
> **État connu :** le refactoring de `src/preprocessing.py` a remplacé `ALL_FEATURES` par `FEATURE_COLS` et `load_raw_data()` par `load_raw(path)`. `src/unsupervised.py` est corrigé depuis le lot 3 (test de non-régression dans [`tests/test_unsupervised.py`](tests/test_unsupervised.py)). `scripts/train_pipeline.py` définit désormais ses propres constantes de compatibilité et régénère `models/best_model.joblib` au format bundle `{model, threshold}`. Voir [`docs/CLUSTERING_AVANCE.md`](docs/CLUSTERING_AVANCE.md).

---

## 9. Couverture du programme du cours

| notions du cours | Statut |
|---|---|
| Pipeline end-to-end, ≥ 10 000 lignes, ≥ 10 attributs | ✅ |
| Types et échelles de mesure, one-hot / ordinal, Z-score | ✅ [`docs/data_dictionary.md`](docs/data_dictionary.md) |
| Outliers IQR + écrêtage | ✅ dataset + notebooks |
| Outliers Z-score, Isolation Forest, LOF | ✅ [`docs/OUTLIERS.md`](docs/OUTLIERS.md), [`scripts/run_outliers.py`](scripts/run_outliers.py), figure [`36`](reports/figures/36_outliers_methodes.png) |
| Similarités euclidienne / Manhattan | ✅ KNN, KMeans |
| Similarité cosinus, Minkowski | ✅ cosinus testée (Minkowski hors périmètre) — [`docs/CLUSTERING_AVANCE.md`](docs/CLUSTERING_AVANCE.md) |
| Classification supervisée, CV 5 plis, métriques | ✅ |
| K-NN, arbres, forêts, Naive Bayes, SVM | ✅ |
| Ensembles (Extra Trees, AdaBoost, Bagging), K-NN cosinus | ✅ [`src/models.py`](src/models.py), 14 configurations définies / 9 variantes évaluées |
| Sélection de variables dans le Pipeline (CV sans fuite) | ✅ `SelectKBest` dans [`src/models.py`](src/models.py) |
| Explicabilité locale SHAP (top 3 facteurs) | ✅ [`src/explainability.py`](src/explainability.py), [`docs/GUIDE_IMPLEMENTATION_MODELE.md`](docs/GUIDE_IMPLEMENTATION_MODELE.md), figure [`42`](reports/figures/42_shap_local.png) |
| K-Medoids (PAM), DIANA, dendrogramme | ✅ [`docs/CLUSTERING_AVANCE.md`](docs/CLUSTERING_AVANCE.md), figures [`39`](reports/figures/39_kmedoids_diana.png) et [`38`](reports/figures/38_dendrogramme.png) |
| Choix de k pour KMeans (coude / silhouette) | ✅ k = 3 retenu — [`docs/CLUSTERING_AVANCE.md`](docs/CLUSTERING_AVANCE.md), figure [`37`](reports/figures/37_choix_k.png) |
| Silhouette | ✅ |
| Davies-Bouldin, indice de Dunn | ✅ [`docs/CLUSTERING_AVANCE.md`](docs/CLUSTERING_AVANCE.md), figure [`40`](reports/figures/40_indices_clustering.png) |
| Règles d'association (support, confiance) | ❌ hors périmètre du projet |
| CRISP-DM en 6 étapes | ✅ §3 |
| Déploiement `joblib`, application web | ✅ [`app/Home.py`](app/Home.py) — `streamlit run app/Home.py` (lot 6) |
| Surveillance / concept drift | ⚠️ documenté comme limite, non implémenté |
| Rapport scientifique PDF | ✅ [`docs/Rapport_MiniProject.pdf`](docs/Rapport_MiniProject.pdf) (15 pages, lots 0 à 4) et [`docs/Rapport_Projet.pdf`](docs/Rapport_Projet.pdf) |

Livrables du lot 3 — [`docs/CLUSTERING_AVANCE.md`](docs/CLUSTERING_AVANCE.md) et [`results/clustering_advanced.csv`](results/clustering_advanced.csv) :

- [`37_choix_k.png`](reports/figures/37_choix_k.png) — inertie, silhouette et Davies-Bouldin selon k ;
- [`38_dendrogramme.png`](reports/figures/38_dendrogramme.png) — dendrogramme de Ward sur un échantillon de 2 000 lignes ;
- [`39_kmedoids_diana.png`](reports/figures/39_kmedoids_diana.png) — partitions K-Medoids et DIANA en projection PCA ;
- [`40_indices_clustering.png`](reports/figures/40_indices_clustering.png) — comparaison des indices pour les sept méthodes ;
- [`41_clusters_cosinus.png`](reports/figures/41_clusters_cosinus.png) — K-Means en géométrie cosinus.

Livrables du lot 4 — [`src/models.py`](src/models.py), [`src/explainability.py`](src/explainability.py), [`results/explainability_top3.csv`](results/explainability_top3.csv) :

- [`42_shap_local.png`](reports/figures/42_shap_local.png) — contribution SHAP moyenne des facteurs locaux sur 200 employés ;
- les six clés du guide (`ZeroR`, `KNN`, `DecisionTree`, `RandomForest`, `NaiveBayes`, `SVM`) restent inchangées dans [`scripts/train_pipeline.py`](scripts/train_pipeline.py), dont les imports ont été réparés ;
- `pytest tests -q` : **79 tests verts** (ajouts successifs pour les modèles étendus, l’explicabilité et l’interface).

Livrables du lot 5 — [`docs/rapport/rapport.md`](docs/rapport/rapport.md), [`scripts/generate_rapport.py`](scripts/generate_rapport.py) :

- [`docs/Rapport_MiniProject.pdf`](docs/Rapport_MiniProject.pdf) — rapport scientifique de 15 pages intégrant les résultats des lots 0 à 4, 16 figures et les intervalles de confiance ;
- [`docs/Rapport_Projet.pdf`](docs/Rapport_Projet.pdf) — copie d'alias maintenue à jour pour les liens existants ;
- le modèle de 47 pages [`Rapport_MiniProject_apercu.pdf`](docs/Rapport_MiniProject_apercu.pdf) est conservé tel quel.

---

## 10. Anonymat, éthique et limites

- **Anonymat :** seuls des codes étudiants sont autorisés dans les livrables. Aucun nom, prénom, matricule ni e-mail personnel n'apparaît dans les fichiers texte suivis du dépôt (vérification effectuée). Les commits restent attribués à chaque compte GitHub individuel, comme demandé pour le suivi des contributions.
- **Éthique :** jeu synthétique, aucune donnée personnelle réelle ; recommandations RH non automatisées ; biais algorithmiques et discrimination involontaire à surveiller.
- **Limites :** déséquilibre 9 %, F1 modeste, signal prédictif faible, clustering peu séparé, données statiques sans détection de dérive conceptuelle.

---

## 11. Structure du dépôt

```text
.
├── README.md, DISCOVERY.md
├── data/raw/                  # CSV source (10 000 × 14)
├── data/processed/            # .gitkeep + données générées (ignorées)
├── notebooks/                 # 5 notebooks
├── src/                       # preprocessing, models, evaluation, stats_tests, unsupervised…
├── scripts/                   # run_preprocessing, train_pipeline, train_models, run_unsupervised, run_outliers…
├── models/                    # .gitkeep + .joblib générés (ignorés)
├── results/                   # CSV versionnés + JSON générés (ignorés)
├── reports/figures/           # 42 figures
├── docs/                      # guide, dictionnaire de données, figures
├── app/                       # application Streamlit (interface)
└── tests/                     # suite pytest
```

## 12. Références

- Support de cours *Introduction au Data Mining*, 2026/2027 — Ph.D. Hosni Adil Imad Eddine (citation bibliographique, aucune coordonnée personnelle reproduite).
- Pedregosa et al., *Scikit-learn: Machine Learning in Python*, JMLR 12:2825-2830, 2011.
- Guide interne [`docs/GUIDE_IMPLEMENTATION_MODELE.md`](docs/GUIDE_IMPLEMENTATION_MODELE.md).
