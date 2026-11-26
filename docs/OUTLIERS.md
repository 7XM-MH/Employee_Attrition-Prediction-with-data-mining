# Analyse des outliers — LOT 2

> Prétraitement avancé : **Z-score**, **IQR de Tukey**, **Isolation Forest** et
> **Local Outlier Factor (LOF)**. Ce document complète [`DISCOVERY.md`](../DISCOVERY.md)
> et la section 9 du [`README.md`](../README.md).

---

## 1. Périmètre et règle de non-fuite

- Les 4 méthodes sont appliquées au **jeu brut** (`data/raw/Employee_Attrition_DataSet.csv`,
  10 000 × 14) à titre **descriptif**.
- Variables examinées : les 5 quantitatives (`Age`, `YearsAtCompany`, `MonthlyIncome`,
  `DistanceFromHome`, `TrainingHoursLastYear`) et les 4 ordinales (`JobLevel`,
  `JobSatisfaction`, `WorkLifeBalance`, `PerformanceRating`).
- **`EmployeeID` et `Attrition` ne sont jamais utilisés** pour détecter un outlier :
  ni comme variable, ni comme critère d'interprétation. Aucune cible n'a servi au calcul.
- Aucun modèle de prédiction, aucune métrique et aucun artefact du pipeline ML
  n'est modifié par ce lot. Le rapport est calculé sur le jeu complet ; si une
  détection devait alimenter le modèle, ses statistiques devraient être apprises
  sur le **train uniquement**, puis appliquées au test.

```bash
python scripts/run_outliers.py
# écrit results/outliers_report.csv et reports/figures/36_outliers_methodes.png
```

---

## 2. Méthodes, formules et seuils retenus

| Méthode | Formule / principe | Seuil retenu | Portée | Traitement recommandé |
|---|---|---|---|---|
| **Z-score** | `z = (x − μ) / σ` (écart-type `ddof=0`) | `|z| > 3` | Univariée | Winsorisation après vérification manuelle |
| **IQR (Tukey)** | Hors de `[Q1 − k·IQR ; Q3 + k·IQR]`, `IQR = Q3 − Q1` | `k = 1,5` | Univariée | Winsorisation ; le pipeline utilise déjà `IQRClipper(factor=3.0)` pour les valeurs extrêmes |
| **Isolation Forest** | Isolement aléatoire dans un espace standardisé ; score anomaly | `contamination = 0,05`, `random_state = 42`, 100 arbres | Multivariée (9 variables) | Investigation ; suppression uniquement si erreur avérée |
| **LOF** | Densité locale par rapport aux `k` voisins les plus proches | `n_neighbors = 20`, `contamination = 0,05` | Multivariée (9 variables) | Investigation ; suppression uniquement si erreur avérée |

L'IQR n'est **pas réimplémentée** dans ce lot : `src/outliers.py` délègue à
`src.preprocessing.iqr_outlier_report(df, cols, factors=(1.5,))`. La winsorisation
historique du notebook 01 et de `IQRClipper` reste inchangée ; ce lot ajoute les
trois méthodes manquantes et harmonise les rapports.

Pour Isolation Forest et LOF, les variables sont imputées (médiane) puis
standardisées sur les données fournies. Sur le jeu réel, `n_neighbors = 20`
est conservé ; sur un petit jeu synthétique il est ramené à `n − 1` pour éviter
une erreur scikit-learn.

---

## 3. Résultats chiffrés (`results/outliers_report.csv`)

Le CSV contient **20 lignes** : 9 variables × 2 méthodes univariées + 2 lignes
multivariées. Colonnes : `methode`, `variable`, `n_outliers`, `pct_outliers`,
`seuil`, `action_recommandee`.

### 3.1 Synthèse par méthode

| Méthode | Effectifs signalés | Lecture |
|---|---:|---|
| Z-score (`|z| > 3`) | **271** occurrences de variables | 2,71 % du jeu en cumul variable ; aucune valeur dans `Age`, `JobLevel`, `JobSatisfaction`, `WorkLifeBalance`, `PerformanceRating` |
| IQR x1,5 | **975** occurrences de variables | 9,75 % en cumul ; dominée par `JobSatisfaction` (500) et `DistanceFromHome` (306) |
| Isolation Forest | **500** lignes | 5,00 % par construction (`contamination = 0,05`) |
| LOF | **500** lignes | 5,00 % par construction ; 210 lignes sont aussi signalées par Isolation Forest |

L'union des deux méthodes multivariées couvre **790 lignes** sur 10 000
(7,90 %) : les deux détecteurs ne se recouvrent que partiellement, ce qui
confirme qu'ils mesurent des aspects différents (isolement global vs densité
locale). Ces totaux sont des **occurrences**, pas des employés uniques supprimés.

### 3.2 Détail complet

| Méthode | Variable | n_outliers | pct_outliers | Seuil |
|---|---|---:|---:|---|
| Z-score | Age | 0 | 0,00 % | `|z| > 3` |
| Z-score | YearsAtCompany | 83 | 0,83 % | `|z| > 3` |
| Z-score | MonthlyIncome | 16 | 0,16 % | `|z| > 3` |
| Z-score | DistanceFromHome | 137 | 1,37 % | `|z| > 3` |
| Z-score | TrainingHoursLastYear | 35 | 0,35 % | `|z| > 3` |
| Z-score | JobLevel | 0 | 0,00 % | `|z| > 3` |
| Z-score | JobSatisfaction | 0 | 0,00 % | `|z| > 3` |
| Z-score | WorkLifeBalance | 0 | 0,00 % | `|z| > 3` |
| Z-score | PerformanceRating | 0 | 0,00 % | `|z| > 3` |
| IQR | Age | 0 | 0,00 % | `IQR x1,5` |
| IQR | YearsAtCompany | 142 | 1,42 % | `IQR x1,5` |
| IQR | MonthlyIncome | 12 | 0,12 % | `IQR x1,5` |
| IQR | DistanceFromHome | 306 | 3,06 % | `IQR x1,5` |
| IQR | TrainingHoursLastYear | 15 | 0,15 % | `IQR x1,5` |
| IQR | JobLevel | 0 | 0,00 % | `IQR x1,5` |
| IQR | JobSatisfaction | 500 | 5,00 % | `IQR x1,5` |
| IQR | WorkLifeBalance | 0 | 0,00 % | `IQR x1,5` |
| IQR | PerformanceRating | 0 | 0,00 % | `IQR x1,5` |
| Isolation Forest | multivarié (9 variables) | 500 | 5,00 % | `contamination=0,05, random_state=42` |
| LOF | multivarié (9 variables) | 500 | 5,00 % | `n_neighbors=20, contamination=0,05` |

Figure associée : [`reports/figures/36_outliers_methodes.png`](../reports/figures/36_outliers_methodes.png).

---

## 4. Décision : winsorisation plutôt que suppression

1. **Aucune suppression automatique.** Un écart statistique n'est pas une erreur.
   Les 500 lignes signalées par Isolation Forest ou LOF sont conservées et
   proposées à l'investigation manuelle.
2. **Winsorisation (écrêtage)** pour les variables continues : le pipeline
   existant `build_preprocessor()` applique déjà `IQRClipper(factor=3.0)`,
   ajusté sur le train uniquement. Ce lot ne modifie pas ce réglage : le
   facteur 1,5 sert au **signalement descriptif**, le facteur 3,0 au seul
   écrêtage des valeurs réellement extrêmes dans le modèle.
3. **Binning** envisageable pour les échelles ordinales (ex. `JobSatisfaction`
   à 5 modalités signalée 500 fois par IQR x1,5) : les bornes 1–5 ont un sens
   métier, il ne faut pas les écrêter comme une grandeur continue.
4. **Suppression uniquement si erreur avérée** (valeur hors plage physique,
   doublon d'identifiant, saisie impossible). `clean_data()` traite déjà les
   hors-plages en les passant à `NaN` ; ce lot n'ajoute aucune règle de suppression.

---

## 5. Mise en garde — outlier ≠ erreur

Un outlier peut être un cas intéressant : fraude, intrusion, panne rare, profil
atypique mais réel. Dans un contexte RH, supprimer automatiquement de tels cas
risquerait d'effacer précisément les situations que l'on cherche à comprendre.
Les méthodes multivariées produisent des **signaux**, pas des verdicts ; la
décision finale reste humaine et documentée.

---

## 6. Couverture du programme

| Notion du cours | État après ce lot |
|---|---|
| Règle de Tukey IQR x1,5 + écrêtage (winsorisation) | ✅ `src/preprocessing.py`, `src/outliers.py` |
| Z-score `|z| > 3` | ✅ `zscore_outliers` |
| Isolation Forest multivarié | ✅ `isolation_forest_outliers` |
| Local Outlier Factor | ✅ `lof_outliers` |
| Traitement par binning / winsorisation / suppression si erreur | ✅ décision documentée en §4 |
| Aucune fuite de données | ✅ rapport descriptif sur brut, pipeline inchangé et ajusté sur le train |
