# Dictionnaire de données — Employee Attrition Dataset

> Référence unique du schéma pour le projet *Employee Attrition Prediction with Data Mining*.
> Toute évolution du schéma doit être répercutée ici **et** dans `src/preprocessing.py`.

## 1. Fiche d'identité

| Propriété | Valeur |
|---|---|
| Fichier | `data/raw/Employee_Attrition_DataSet.csv` |
| Encodage | UTF-8 |
| Séparateur | virgule (`,`) |
| Dimensions | **10 000 lignes × 14 colonnes** |
| Variables explicatives | **12** (9 numériques + 3 catégorielles) |
| Identifiant | `EmployeeID` — 1 colonne, exclue des features |
| Variable cible | `Attrition` (binaire) |
| Valeurs manquantes | **0** |
| Doublons de lignes | **0** |
| Taille du fichier | ≈ 514 Ko |
| Nature | Dataset **synthétique**, usage académique uniquement |

### Ordre exact des colonnes dans le fichier

```text
EmployeeID, Age, Department, JobLevel, YearsAtCompany, MonthlyIncome,
JobSatisfaction, WorkLifeBalance, OverTime, DistanceFromHome,
PromotionLast5Years, PerformanceRating, TrainingHoursLastYear, Attrition
```

Cet ordre est un invariant : la prédiction en lot (`app/pages/5_Prediction_en_lot.py`)
et `src/preprocessing.py` sélectionnent les colonnes **par nom**, mais tout export
doit conserver cette liste pour rester lisible.

---

## 2. Variable cible — `Attrition`

| Classe | Effectif | Pourcentage | Encodage retenu |
|---|---:|---:|---:|
| `No` (reste dans l'entreprise) | 9 097 | 90,97 % | `0` |
| `Yes` (quitte l'entreprise) | 903 | 9,03 % | `1` |
| **Total** | **10 000** | **100 %** | — |

**Déséquilibre : ratio ≈ 10,07 : 1.**

Conséquences méthodologiques :

- L'*accuracy* est trompeuse : un modèle qui prédit toujours `No` atteint déjà **90,97 %**.
- Les métriques de pilotage sont **F1, recall, precision, ROC-AUC et PR-AUC**, avec un
  intérêt particulier pour la classe minoritaire `Yes`.
- Le déséquilibre doit être traité **dans la validation croisée** (`class_weight`,
  stratégie de rééchantillonnage, réglage du seuil) et **jamais avant le split**
  train/test, sous peine de fuite de données.
- Le mapping `No → 0`, `Yes → 1` est défini dans `src/preprocessing.py`
  (`prepare_data`) et doit être identique partout (entraînement, application, tests).

---

## 3. Dictionnaire des variables

Légende des rôles :

- **ID** : identifiant, exclu des variables explicatives.
- **Num.** : variable numérique.
- **Cat.** : variable catégorielle (traitée par one-hot encoding).
- **Cible** : variable à prédire.

| # | Colonne | Type pandas | Type métier | Plage / modalités | Rôle | Traitement prévu |
|---:|---|---|---|---|---|---|
| 1 | `EmployeeID` | `str` | Identifiant | `EMP0000` … `EMP9999` (10 000 valeurs uniques) | **ID** | **Exclu de `X`** : aucun pouvoir prédictif, risque de fuite. |
| 2 | `Age` | `int64` | Num. continue | 22 – 60 ans | Num. | `StandardScaler` |
| 3 | `Department` | `str` | Cat. nominale | `Sales`, `Engineering`, `Operations`, `Marketing`, `Finance`, `IT`, `HR` (7 modalités) | Cat. | `OneHotEncoder(handle_unknown="ignore")` |
| 4 | `JobLevel` | `int64` | Num. ordinale | 1 – 5 | Num. | `StandardScaler` |
| 5 | `YearsAtCompany` | `int64` | Num. discrète | 0 – 35 ans | Num. | `StandardScaler` |
| 6 | `MonthlyIncome` | `int64` | Num. continue | 3 000 – 21 410 (unité monétaire non précisée) | Num. | `StandardScaler` |
| 7 | `JobSatisfaction` | `int64` | Num. ordinale | 1 – 5 | Num. | `StandardScaler` |
| 8 | `WorkLifeBalance` | `int64` | Num. ordinale | 1 – 4 | Num. | `StandardScaler` |
| 9 | `OverTime` | `str` | Cat. binaire | `No`, `Yes` | Cat. | `OneHotEncoder` |
| 10 | `DistanceFromHome` | `float64` | Num. continue | 1,0 – 50,0 (valeurs décimales) | Num. | `StandardScaler` |
| 11 | `PromotionLast5Years` | `str` | Cat. binaire | `No`, `Yes` | Cat. | `OneHotEncoder` |
| 12 | `PerformanceRating` | `int64` | Num. ordinale | 1 – 4 | Num. | `StandardScaler` |
| 13 | `TrainingHoursLastYear` | `int64` | Num. discrète | 2 – 31 heures | Num. | `StandardScaler` |
| 14 | `Attrition` | `str` | **Cible** binaire | `No`, `Yes` | **Cible** | Mapping `No → 0`, `Yes → 1` |

### Choix de modélisation à documenter

Les variables `JobLevel`, `JobSatisfaction`, `WorkLifeBalance` et `PerformanceRating`
sont **ordinales** mais stockées sous forme d'entiers. Le pipeline actuel les traite
comme des variables **numériques standardisées**. Ce choix est volontaire :

- les modèles à base d'arbres n'ont pas besoin d'encodage supplémentaire ;
- la standardisation est sans effet sur les arbres et indispensable pour K-NN et SVM ;
- il évite d'exploser la dimension avec un one-hot sur des échelles de 4–5 niveaux.

Une alternative (one-hot) est possible mais n'est pas retenue pour garder un schéma
unique entre l'entraînement et l'application.

---

## 4. Statistiques descriptives — variables numériques

| Colonne | min | Q1 | médiane | moyenne | Q3 | max | écart-type | valeurs uniques | zéros |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `Age` | 22 | 31,0 | 37,0 | 37,69 | 44,0 | 60 | 9,38 | 39 | 0 |
| `JobLevel` | 1 | 1,0 | 2,0 | 2,38 | 3,0 | 5 | 1,14 | 5 | 0 |
| `YearsAtCompany` | 0 | 2,0 | 6,0 | 7,79 | 12,0 | 35 | 7,14 | 36 | 1 452 |
| `MonthlyIncome` | 3 000 | 6 043,5 | 8 670,0 | 8 967,92 | 11 484,0 | 21 410 | 3 500,75 | 6 784 | 0 |
| `JobSatisfaction` | 1 | 3,0 | 3,0 | 3,36 | 4,0 | 5 | 1,06 | 5 | 0 |
| `WorkLifeBalance` | 1 | 2,0 | 3,0 | 2,76 | 3,0 | 4 | 0,89 | 4 | 0 |
| `DistanceFromHome` | 1,0 | 4,7 | 8,3 | 9,91 | 13,4 | 50,0 | 6,97 | 381 | 0 |
| `PerformanceRating` | 1 | 2,0 | 3,0 | 2,60 | 3,0 | 4 | 0,74 | 4 | 0 |
| `TrainingHoursLastYear` | 2 | 12,0 | 15,0 | 14,99 | 18,0 | 31 | 3,89 | 30 | 0 |

Observations utiles :

- `YearsAtCompany = 0` pour **1 452 employés** (14,52 %) : ce sont des arrivées
  récentes, pas des valeurs manquantes. Ne pas les imputer ni les supprimer.
- `DistanceFromHome` est la seule variable numérique **non entière** : elle doit être
  lue en `float` (le CSV contient des valeurs comme `4.7`).
- `MonthlyIncome` a 6 784 valeurs distinctes sur 10 000 lignes : distribution quasi
  continue, sans valeur sentinelle (aucun 0, aucun -1).

---

## 5. Modalités des variables catégorielles

| Colonne | Modalité | Effectif | Part | Taux d'attrition observé* |
|---|---|---:|---:|---:|
| `Department` | `Sales` | 2 453 | 24,53 % | 9,50 % |
| | `Engineering` | 2 045 | 20,45 % | 9,58 % |
| | `Operations` | 1 513 | 15,13 % | 8,92 % |
| | `Marketing` | 1 188 | 11,88 % | 9,09 % |
| | `Finance` | 1 037 | 10,37 % | 8,20 % |
| | `IT` | 1 035 | 10,35 % | 8,99 % |
| | `HR` | 729 | 7,29 % | 7,27 % |
| `OverTime` | `No` | 7 151 | 71,51 % | 7,78 % |
| | `Yes` | 2 849 | 28,49 % | **12,18 %** |
| `PromotionLast5Years` | `No` | 9 086 | 90,86 % | 9,27 % |
| | `Yes` | 914 | 9,14 % | 6,67 % |

\* Taux indicatif, descriptif uniquement : il ne constitue **pas** un test
statistique. Les hypothèses H1–H4 et leur significativité sont traitées dans le
notebook d'analyse statistique (`notebooks/02_statistical_analysis.ipynb`).

Lecture prudente : `OverTime = Yes` présente un taux d'attrition environ 1,6 fois
supérieur à `OverTime = No`. Cette piste devra être confirmée par un test du χ² et
une taille d'effet (Cramér's V), pas interprétée comme une relation causale.

---

## 6. Contrôles qualité

| Contrôle | Résultat | Statut |
|---|---|---|
| Valeurs manquantes (toutes colonnes) | 0 | ✅ |
| Doublons de lignes | 0 | ✅ |
| `EmployeeID` unique | oui (10 000 / 10 000) | ✅ |
| `EmployeeID` conforme au motif `EMP\d{4}` | oui | ✅ |
| Valeurs de la cible ∈ {`No`, `Yes`} | oui | ✅ |
| Effectifs de la cible | 9 097 / 903 | ✅ (déséquilibre documenté) |
| Âge dans une plage plausible | 22 – 60 | ✅ |
| `MonthlyIncome` > 0 | oui (min 3 000) | ✅ |
| Aucune valeur négative | oui | ✅ |
| Toutes les variables explicatives présentes | oui (12 / 12) | ✅ |
| `EmployeeID` absent des features | oui | ✅ |
| Cible absente des features | oui | ✅ |

### Cas particuliers à ne pas « corriger » aveuglément

1. `YearsAtCompany = 0` (1 452 lignes) : ancienneté nulle, valeur métier légitime.
2. `DistanceFromHome` décimal : unité non documentée, mais cohérente (km probable).
3. Aucun outlier n'est supprimé : les extrêmes (`MonthlyIncome` à 21 410,
   `YearsAtCompany` à 35) restent plausibles pour un dataset synthétique. Une
   suppression changerait la distribution sans justification métier.
4. Aucune imputation n'est nécessaire aujourd'hui, mais le pipeline doit rester
   robuste si le CSV évolue (voir `docs/GUIDE_IMPLEMENTATION_MODELE.md`).

---

## 7. Décisions de prétraitement

1. **Exclure `EmployeeID`** de `X` (identifiant, pas un prédicteur).
2. **Encoder la cible** `No → 0`, `Yes → 1` via `prepare_data`.
3. **Ne pas imputer** : 0 valeur manquante sur les 10 000 lignes.
4. **Ne pas dédupliquer** : 0 doublon détecté (contrôle conservé dans les tests).
5. **Standardiser** les 9 variables numériques (`StandardScaler`).
6. **One-hot encoder** les 3 variables catégorielles
   (`OneHotEncoder(handle_unknown="ignore")` pour accepter une modalité inconnue
   en production sans lever d'erreur).
7. **Ajuster le préprocesseur sur le train uniquement**, puis transformer le test
   (`Pipeline` + `ColumnTransformer`) : aucune fuite de données.
8. **Traiter le déséquilibre après le split**, dans la CV : `class_weight="balanced"`,
   pondération ou réglage du seuil de décision. Jamais de SMOTE avant le split.
9. **Sélectionner le modèle sur la validation croisée** (F1 moyen), le jeu de test
   ne servant qu'à l'évaluation finale.

---

## 8. Contrat d'interface

`src/preprocessing.py` est la **source de vérité** du schéma :

```python
NUMERICAL_FEATURES = [
    "Age", "JobLevel", "YearsAtCompany", "MonthlyIncome",
    "JobSatisfaction", "WorkLifeBalance", "DistanceFromHome",
    "PerformanceRating", "TrainingHoursLastYear",
]
CATEGORICAL_FEATURES = ["Department", "OverTime", "PromotionLast5Years"]
TARGET = "Attrition"
ID_COL = "EmployeeID"
ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES  # 12 colonnes
```

Invariants attendus par les consommateurs :

| Producteur | Consommateur | Contrat |
|---|---|---|
| `load_raw_data()` | notebooks, tests | DataFrame de forme `(10000, 14)`, colonnes dans l'ordre du fichier. |
| `prepare_data(df)` | entraînement, notebooks | `X` : DataFrame `(n, 12)` avec `ALL_FEATURES` uniquement ; `y` : Series `int` de 0/1. |
| `models/best_model.joblib` | application Streamlit | `Pipeline` complet ; `predict_proba(X)[:, 1]` = probabilité de `Yes`. |
| Prédiction en lot | `app/pages/5_Prediction_en_lot.py` | Le CSV d'entrée doit contenir les 12 colonnes ; `EmployeeID` et `Attrition` sont optionnels/ignorés. |

---

## 9. Provenance et usage

- Dataset **synthétique** fourni pour le mini-projet de Data Mining (Track A :
  classification supervisée). Il ne décrit aucune personne réelle.
- **Usage académique uniquement.** Les prédictions ne doivent jamais servir de
  base à une décision RH réelle (licenciement, promotion, sanction).
- Les résultats produits sur ce jeu de données ne sont pas généralisables à une
  population réelle : la distribution et le déséquilibre sont simulés.

---

*Dernière vérification : 2026-11-09, sur `data/raw/Employee_Attrition_DataSet.csv`
(commit de référence `main` = `2642f11`). Toutes les valeurs de ce document ont été
recalculées à partir du fichier, pas recopiées.*
