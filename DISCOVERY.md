# Découverte du Jeu de Données — Employee Attrition

## Vue d'ensemble

| Propriété           | Valeur                                      |
|---------------------|---------------------------------------------|
| **Fichier**         | `data/raw/Employee_Attrition_DataSet.csv`   |
| **Lignes**          | 10 000                                      |
| **Colonnes**        | 14                                          |
| **Valeurs manquantes** | 0                                        |
| **Doublons**        | 0                                           |
| **Variable cible**  | `Attrition` (No: 9 097 / Yes: 903)         |

> [!IMPORTANT]
> Le jeu de données est **déséquilibré** : seulement ~9 % des employés ont quitté l'entreprise (`Yes`).

---

## Description des Colonnes

| Colonne               | Type    | Plage / Modalités                                                                 | Rôle            |
|------------------------|---------|-----------------------------------------------------------------------------------|-----------------|
| `EmployeeID`           | object  | EMP0000 → EMP9999                                                                 | Identifiant (à supprimer) |
| `Age`                  | int64   | 22 – 60                                                                           | Numérique       |
| `Department`           | object  | Sales (2453), Engineering (2045), Operations (1513), Marketing (1188), Finance (1037), IT (1035), HR (729) | Catégorielle |
| `JobLevel`             | int64   | 1 – 5                                                                             | Numérique       |
| `YearsAtCompany`       | int64   | 0 – 35                                                                            | Numérique       |
| `MonthlyIncome`        | int64   | 3 000 – 21 410                                                                    | Numérique       |
| `JobSatisfaction`      | int64   | 1 – 5                                                                             | Numérique       |
| `WorkLifeBalance`      | int64   | 1 – 4                                                                             | Numérique       |
| `OverTime`             | object  | No (7 151), Yes (2 849)                                                           | Catégorielle    |
| `DistanceFromHome`     | float64 | 1 – 50                                                                            | Numérique       |
| `PromotionLast5Years`  | object  | No (9 086), Yes (914)                                                             | Catégorielle    |
| `PerformanceRating`    | int64   | 1 – 4                                                                             | Numérique       |
| `TrainingHoursLastYear`| int64   | 2 – 31                                                                            | Numérique       |
| `Attrition`            | object  | No (9 097), Yes (903)                                                             | **Cible**       |

### Résumé des types

**Selon le type pandas brut :**

- **Numériques (9)** : Age, JobLevel, YearsAtCompany, MonthlyIncome, JobSatisfaction, WorkLifeBalance, DistanceFromHome, PerformanceRating, TrainingHoursLastYear
- **Catégorielles (3)** : Department, OverTime, PromotionLast5Years

**Selon l'échelle de mesure (celle utilisée par `src/preprocessing.py`) :**

- **Quantitatives (5)** : Age, YearsAtCompany, MonthlyIncome, DistanceFromHome, TrainingHoursLastYear → `StandardScaler`
- **Ordinales (4)** : JobLevel, JobSatisfaction, WorkLifeBalance, PerformanceRating → ordre conservé
- **Binaires Yes/No (2)** : OverTime, PromotionLast5Years → `0/1`
- **Nominale (1)** : Department → `OneHotEncoder(handle_unknown="ignore")`

Le détail des échelles (nominale, ordinale, intervalle, ratio) est dans [`docs/data_dictionary.md`](docs/data_dictionary.md).

---

## Déséquilibre de Classes

```
No  ████████████████████████████████████████████████  9 097  (90.97 %)
Yes █████                                               903  ( 9.03 %)
```

Le ratio est d'environ **10:1**. La métrique F1-score sera privilégiée pour l'évaluation des modèles afin de mieux capturer la capacité de détection de la classe minoritaire.

---

## Contrôles qualité et analyses disponibles

| Élément | Emplacement |
|---|---|
| Statistiques descriptives (moyennes, quartiles, IQR, asymétrie) | [`results/descriptive_stats.csv`](results/descriptive_stats.csv) |
| Tests vs attrition (Mann-Whitney, χ², taille d'effet, p ajustée BH) | [`results/stat_tests_vs_attrition.csv`](results/stat_tests_vs_attrition.csv) |
| Corrélations de Spearman | [`results/stat_spearman_paires.csv`](results/stat_spearman_paires.csv) |
| Tests de normalité (Shapiro-Wilk) | [`results/stat_normalite_shapiro.csv`](results/stat_normalite_shapiro.csv) |
| Outliers IQR et écrêtage | [`notebooks/01_eda_and_cleaning.ipynb`](notebooks/01_eda_and_cleaning.ipynb), `src/preprocessing.py` |
| Figures EDA (distributions, corrélations, taux de départ) | [`reports/figures/`](reports/figures/) |

Rappel : les outliers sont détectés par la règle IQR × 1,5 et écrêtés (winsorisation). Le jeu brut ne contient ni valeur manquante ni doublon.

---

## État des artefacts

Les sources sont versionnées ; les artefacts générés ne le sont pas (`.gitignore`).

| Catégorie | Versionné | Généré par |
|---|---|---|
| Jeu brut `data/raw/*.csv` | ✅ | — |
| Données traitées `data/processed/*.csv` | ❌ | `python scripts/run_preprocessing.py` |
| Modèles `models/*.joblib` | ❌ | `scripts/train_pipeline.py`, `scripts/train_models.py` |
| Résultats JSON `results/*.json` | ❌ | scripts d'entraînement et `run_unsupervised.py` |
| Résultats CSV `results/*.csv` | ✅ | scripts d'entraînement |
| Figures `reports/figures/*.png` | ✅ | notebooks / scripts |
| Notebooks `notebooks/*.ipynb` | ✅ | — |

---

## Conclusion

Le jeu est propre, déséquilibré (9,03 % de `Yes`) et documenté. Le F1-score est privilégié pour l'évaluation. Les commandes de reproduction et les résultats chiffrés sont dans le [`README.md`](README.md).

> [!NOTE]
> Le refactoring de `src/preprocessing.py` a renommé quelques symboles (`ALL_FEATURES` → `FEATURE_COLS`, `load_raw_data()` → `load_raw(path)`). Un correctif de compatibilité pour `scripts/train_pipeline.py`, `src/unsupervised.py` et les tests concernés est prévu dans une PR séparée.
