# Dictionnaire de données — Employee Attrition Dataset

**Fichier :** `data/raw/employee_attrition_2026.csv`
**Encodage :** UTF-8
**Séparateur :** virgule
**Dimensions :** 59 598 lignes × 24 colonnes
**Valeurs manquantes :** 0
**Doublons :** 0

La variable cible est `Attrition` (classification binaire) :

| Classe | Effectif | Pourcentage |
|---|---:|---:|
| `Stayed` | 31 260 | 52,45 % |
| `Left` | 28 338 | 47,55 % |

---

## Variables

| # | Colonne | Type | Valeurs / plage | Rôle | Remarques |
|---:|---|---|---|---|---|
| 1 | `Employee ID` | Entier | 1 – 74 498 (59 598 valeurs uniques) | Identifiant | **À exclure des variables explicatives** (aucun pouvoir prédictif, risque de fuite). |
| 2 | `Age` | Entier | 18 – 59 | Numérique | Âge en années. |
| 3 | `Gender` | Catégorielle | `Male`, `Female` | Nominale | — |
| 4 | `Years at Company` | Entier | 1 – 51 | Numérique | Ancienneté en années. |
| 5 | `Job Role` | Catégorielle | `Technology`, `Healthcare`, `Education`, `Media`, `Finance` | Nominale | — |
| 6 | `Monthly Income` | Entier | 1 316 – 16 149 | Numérique | Revenu mensuel (unité monétaire non précisée). |
| 7 | `Work-Life Balance` | Catégorielle | `Excellent`, `Good`, `Fair`, `Poor` | Ordinale | Ordre : Poor < Fair < Good < Excellent. |
| 8 | `Job Satisfaction` | Catégorielle | `Very High`, `High`, `Medium`, `Low` | Ordinale | Ordre : Low < Medium < High < Very High. |
| 9 | `Performance Rating` | Catégorielle | `High`, `Average`, `Below Average`, `Low` | Ordinale | Ordre : Low < Below Average < Average < High. |
| 10 | `Number of Promotions` | Entier | 0 – 4 | Numérique | Nombre de promotions reçues. |
| 11 | `Overtime` | Catégorielle | `Yes`, `No` | Binaire | Heures supplémentaires. |
| 12 | `Distance from Home` | Entier | 1 – 99 | Numérique | Distance domicile-travail (unité non précisée). |
| 13 | `Education Level` | Catégorielle | `PhD`, `Master’s Degree`, `Bachelor’s Degree`, `Associate Degree`, `High School` | Ordinale | Contient une apostrophe typographique (U+2019) à normaliser. |
| 14 | `Marital Status` | Catégorielle | `Married`, `Single`, `Divorced` | Nominale | — |
| 15 | `Number of Dependents` | Entier | 0 – 6 | Numérique | Personnes à charge. |
| 16 | `Job Level` | Catégorielle | `Entry`, `Mid`, `Senior` | Ordinale | — |
| 17 | `Company Size` | Catégorielle | `Small`, `Medium`, `Large` | Ordinale | — |
| 18 | `Company Tenure` | Entier | 2 – 128 | Numérique | **Unité ambiguë** : très probablement en mois (voir note ci-dessous). |
| 19 | `Remote Work` | Catégorielle | `Yes`, `No` | Binaire | — |
| 20 | `Leadership Opportunities` | Catégorielle | `Yes`, `No` | Binaire | — |
| 21 | `Innovation Opportunities` | Catégorielle | `Yes`, `No` | Binaire | — |
| 22 | `Company Reputation` | Catégorielle | `Excellent`, `Good`, `Fair`, `Poor` | Ordinale | Ordre : Poor < Fair < Good < Excellent. |
| 23 | `Employee Recognition` | Catégorielle | `Very High`, `High`, `Medium`, `Low` | Ordinale | Ordre : Low < Medium < High < Very High. |
| 24 | `Attrition` | Catégorielle | `Stayed`, `Left` | **Cible** | Variable à prédire. |

---

## Contrôles de cohérence

| Contrôle | Résultat | Interprétation |
|---|---:|---|
| `Age` hors de [18, 59] | 0 | OK |
| `Monthly Income` ≤ 0 | 0 | OK |
| `Years at Company` > `Age − 16` | 19 050 (31,96 %) | Incohérence apparente du dataset synthétique : ancienneté impossible avant 16 ans. À documenter, pas nécessairement à corriger. |
| `Company Tenure` > `Age` | 42 283 (70,94 %) | Cohérent si `Company Tenure` est exprimé **en mois** et `Age` **en années**. À clarifier avant tout usage. |
| `Number of Dependents` < 0 | 0 | OK |

---

## Décisions de prétraitement recommandées

1. **Exclure `Employee ID`** du jeu de variables explicatives.
2. **Ne pas imputer** : aucune valeur manquante.
3. **Supprimer les doublons** : aucun doublon détecté (contrôle conservé par sécurité).
4. **Normaliser l'apostrophe** de `Education Level` (`’` → `'`) pour éviter les problèmes d'encodage.
5. **Encoder les variables ordinales** dans leur ordre naturel (work-life balance, satisfaction, performance, éducation, niveau hiérarchique, taille d'entreprise, réputation, reconnaissance).
6. **Encoder en one-hot** les variables nominales (genre, poste, statut marital).
7. **Encoder en binaire** (0/1) les variables `Yes`/`No`.
8. **Standardiser** les variables numériques (`Age`, `Years at Company`, `Monthly Income`, `Number of Promotions`, `Distance from Home`, `Number of Dependents`, `Company Tenure`).
9. **Documenter l'unité de `Company Tenure`** et envisager une transformation (mois → années) si l'unité est confirmée.
10. **Fit du prétraitement sur le train uniquement** (via `Pipeline` / `ColumnTransformer`) pour éviter toute fuite de données.

---

## Source

- Dataset Kaggle : `stealthtechnologies/employee-attrition-dataset`.
- Fichier versionné dans le dépôt : `data/raw/employee_attrition_2026.csv` (7,7 Mo).
- Dataset synthétique à usage académique uniquement.
