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

- **Numériques (9)** : Age, JobLevel, YearsAtCompany, MonthlyIncome, JobSatisfaction, WorkLifeBalance, DistanceFromHome, PerformanceRating, TrainingHoursLastYear
- **Catégorielles (3)** : Department, OverTime, PromotionLast5Years

---

## Déséquilibre de Classes

```
No  ████████████████████████████████████████████████  9 097  (90.97 %)
Yes █████                                               903  ( 9.03 %)
```

Le ratio est d'environ **10:1**. La métrique F1-score sera privilégiée pour l'évaluation des modèles afin de mieux capturer la capacité de détection de la classe minoritaire.

---

## État des Artefacts du Projet

| Élément                          | Statut         |
|----------------------------------|----------------|
| `data/raw/Employee_Attrition_DataSet.csv` | ✅ Présent |
| `data/processed/`                | ⚠️ Vide        |
| `models/`                        | ⚠️ Vide        |
| `results/`                       | ⚠️ Vide        |
| `scripts/train_pipeline.py`      | ⚠️ Stub (0 octets) |
| `src/*.py`                       | ⚠️ Stubs (0 octets) |
| `notebooks/*.ipynb`              | ⚠️ Vides       |

### Artefacts manquants

- Modèles entraînés (`.joblib`)
- Métriques d'évaluation (`metrics.json`)
- Matrices de confusion (`confusion_matrices.json`)
- Importances des caractéristiques (`feature_importances.json`)
- Données ROC (`roc_data.json`)
- Meilleurs hyperparamètres (`best_hyperparameters.json`)
- Résumé de validation croisée (`cv_results_summary.json`)
- Jeu de test sauvegardé (`data/processed/test_set.csv`)

---

## Conclusion

> [!NOTE]
> Le pipeline d'entraînement complet `scripts/train_pipeline.py` doit être exécuté pour générer tous les artefacts nécessaires (modèles, métriques, visualisations). Ce script entraîne 6 modèles (Zero-R, K-NN, Arbre de Décision, Forêt Aléatoire, Naive Bayes, SVM) avec recherche d'hyperparamètres et sauvegarde tous les résultats dans les dossiers `models/` et `results/`.

```bash
python scripts/train_pipeline.py
```
