# Plan de projet — LOT 0 : Organisation et plan de projet

> CRISP-DM étape 1 — compréhension métier et cadrage organisationnel.
> Ce document fixe le calendrier, les rôles, les critères de succès et la traçabilité Git du mini-projet *Employee Attrition*.

---

## 1. Contexte métier

Le renouvellement du personnel entraîne des coûts de recrutement, d'intégration et une perte de compétences. L'objectif pédagogique est d'anticiper les départs à partir de caractéristiques individuelles et organisationnelles, sur un jeu **synthétique** de 10 000 employés.

Aucune prédiction produite par ce dépôt ne doit être utilisée pour une décision RH réelle. Le projet est académique et fictif.

**Question métier :** les variables démographiques, hiérarchiques et d'environnement de travail permettent-elles de prédire l'attrition ?

---

## 2. Objectifs Data Mining

| Objectif | Opérationnalisation |
|---|---|
| Classification supervisée | Cible `Attrition` codée `No → 0`, `Yes → 1` ; classe positive = `Yes` (~9,03 % des lignes) |
| Jeu de données | 10 000 lignes × 14 colonnes ; `EmployeeID` exclu de `X` ; 12 variables explicatives retenues |
| Validation | `StratifiedKFold` avec **K ≥ 5**, shuffle et graine fixe ; jeu de test isolé de tout ajustement |
| Métriques principales | **F1** et **Average Precision (AP)** ; l'accuracy seule est trompeuse (Zero-R atteint déjà 0,9095) |
| Analyse complémentaire | Clustering non supervisé descriptif (`Attrition` jamais utilisée pendant l'ajustement) |
| Reproductibilité | Scripts depuis la racine, artefacts générés ignorés par Git, suite pytest verte |
| Livrable scientifique | Rapport PDF, figures et tableaux de métriques versionnés |

Références de cadrage : [`README.md`](../README.md), [`DISCOVERY.md`](../DISCOVERY.md), [`docs/GUIDE_IMPLEMENTATION_MODELE.md`](GUIDE_IMPLEMENTATION_MODELE.md), [`results/tableau_metriques_rapport.csv`](../results/tableau_metriques_rapport.csv).

---

## 3. Équipe et rôles

Les codes ci-dessous sont des **valeurs par défaut** au format `IASDxx → rôle`, mises à jour par l'équipe après validation. Aucun nom, prénom, matricule ni adresse personnelle n'est enregistré dans le dépôt ; seuls les codes étudiants et les identifiants GitHub de revue sont utilisés.

| Code étudiant | Rôle principal | Lots concernés |
|---|---|---|
| IASD01 | Chef de projet, planning et revue Git | 0, 6 |
| IASD02 | Données, EDA et dictionnaire de données | 1, 2 |
| IASD03 | Modélisation supervisée et sélection CV | 1, 4 |
| IASD04 | Analyse non supervisée et optimisation | 2, 5 |
| IASD05 | Tests, qualité et reproductibilité | 3 |
| IASD06 | Documentation, rapport et métriques | 4, 6 |

La partie interface (`app/**`, `.streamlit/**`, `src/app.py`) appartient à un autre membre et est **hors périmètre** des lots suivants.

---

## 4. Rétroplanning prévisionnel des lots 0 → 6

| Lot | Période prévisionnelle | Objectif | Branche | PR |
|---|---|---|---|---|
| 0 | 24 – 27 nov 2026 | Organisation, rôles, critères de succès, traçabilité | `feature/lot0-organisation` | PR à ouvrir |
| 1 | 28 nov – 04 déc 2026 | Pipeline supervisé conforme au guide, 6 modèles imposés | `feature/lot1-entrainement` | #4 (mergée) |
| 2 | 05 – 11 déc 2026 | Non supervisé + hygiène des artefacts générés | `feature/lot2-unsupervised`, `feature/lot2-gitignore` | #5 et #6 (mergées) |
| 3 | 12 – 18 déc 2026 | Suite de tests et contrats de données | `feature/lot3-tests` | #7 (mergée) |
| 4 | 19 – 25 déc 2026 | Documentation, README, dictionnaire, mapping CRISP-DM | `feature/lot4-documentation` | #8 (mergée) |
| 5 | 26 déc 2026 – 01 janv 2027 | Optimisation, seuil, bootstrap, classement multicritère | `feature/lot5-optimisation` | à créer |
| 6 | 02 – 08 janv 2027 | Rapport scientifique final et clôture | `feature/lot6-rapport` | à créer |

Le cycle est itératif : les résultats du lot 5 peuvent conduire à réajuster la préparation et la modélisation des lots 1 et 2.

---

## 5. Critères de succès mesurables

| Critère | Cible vérifiable |
|---|---|
| Volume de données | 10 000 lignes, 14 colonnes, 0 valeur manquante, 0 doublon |
| Features | 12 variables excluant `EmployeeID` et `Attrition` |
| Encodage de la cible | `No → 0`, `Yes → 1` ; probabilité lue sur la classe `Yes` |
| Validation croisée | `StratifiedKFold(n_splits >= 5, shuffle=True, random_state=...)` |
| Sélection modèle | F1 moyen CV pour le pipeline du guide, AP moyen CV pour l'analyse étendue ; test jamais utilisé pour choisir |
| Métriques reporting | F1, AP, ROC-AUC, matrices de confusion, IC bootstrap |
| Suite de tests | `python -m pytest tests -q` → **40 passed** (tout écart doit être expliqué dans la PR) |
| Livrables graphiques | 5 notebooks, 35 figures versionnées |
| Modèles | 9 modèles supervisés (dont Zero-R), 4 méthodes non supervisées (KMeans, PCA, agglomératif, DBSCAN) |
| Traçabilité | 1 branche `feature/lotN-*`, 1 commit, 1 PR par lot, revue de `Anwar77-b` |
| Anonymat | 0 nom, prénom, matricule ou adresse personnelle dans les fichiers suivis |

---

## 6. Définition de « Done »

Un lot est considéré comme terminé uniquement si :

1. la branche part de `upstream/main` et porte le nom `feature/lotN-*` ;
2. un seul commit conventionnel contient l'ensemble des changements du lot ;
3. une seule PR est ouverte vers `Anwar77-b:main`, avec revue demandée à `Anwar77-b` et commentaire de demande de revue ;
4. l'auteur de la PR ne la merge pas lui-même ;
5. les vérifications passent : pytest, absence de marqueurs temporaires, liens Markdown internes valides, `git diff --check` ;
6. aucun fichier sous `app/**`, `.streamlit/**` ni `src/app.py` n'est modifié ;
7. aucun artefact généré n'est ajouté (`*.joblib`, `results/*.json`, `data/processed/*.csv`) ;
8. la documentation et la matrice de traçabilité sont à jour.

---

## 7. Matrice de traçabilité lot → branche → PR → fichiers → reviewer

| Lot | Branche | PR | Fichiers principaux | Reviewer |
|---|---|---|---|---|
| 0 | `feature/lot0-organisation` | PR à ouvrir | `docs/PLAN_PROJET.md`, `CONTRIBUTING.md`, `LICENSE`, `README.md` | Anwar77-b |
| 1 | `feature/lot1-entrainement` | #4 | `scripts/train_pipeline.py`, `scripts/train_models.py`, `src/models.py`, `results/cv_summary.csv`, `results/model_comparison.csv` | Anwar77-b |
| 2 | `feature/lot2-unsupervised`, `feature/lot2-gitignore` | #5, #6 | `src/unsupervised.py`, `scripts/run_unsupervised.py`, `.gitignore` | Anwar77-b |
| 3 | `feature/lot3-tests` | #7 | `tests/*.py` | Anwar77-b |
| 4 | `feature/lot4-documentation` | #8 | `README.md`, `DISCOVERY.md`, `docs/data_dictionary.md`, `docs/GUIDE_IMPLEMENTATION_MODELE.md` | Anwar77-b |
| 5 | `feature/lot5-optimisation` | à créer | `src/optimization.py`, `scripts/run_optimization.py`, `results/optimization/*.csv` | Anwar77-b |
| 6 | `feature/lot6-rapport` | à créer | `docs/Rapport_Projet.pdf`, `docs/figures/*.png`, `results/tableau_metriques_rapport.csv` | Anwar77-b |

Action hors lot réalisée pendant le LOT 0 : la PR #2, doublon de la PR #3 déjà mergée, est fermée avec un commentaire poli, **sans merge**.

---

## 8. Objectif de note et écart constaté

Grille reçue (situation avant ce lot) :

|Rubrique | Note | Commentaire |
|---|---:|---|
| Technical ML Core | 3,6 / 10 | à consolider par les lots 5 et 6 |
| Application & UI | 1,6 / 4 | partie interface, hors périmètre de notre travail |
| Git Traceability | 1,2 / 3 | objectif : 3 / 3 grâce à une branche et une PR par lot |
| Documentation | 0,1 / 3 | objectif : 3 / 3 grâce à ce plan et au rapport |
| **Note globale** | **6,5 / 20** | pénalité de **-0,70 pt** pour rapport PDF scientifique manquant |

Objectif : **16 / 20 au maximum sans l'interface**, **20 / 20 avec l'interface** livrée par l'autre membre. Le rapport `docs/Rapport_Projet.pdf` est désormais présent sur `upstream/main` (commit `9233737`) ; la levée effective de la pénalité reste à confirmer par le professeur.

---

## 9. hors périmètre

Ne sont pas traités dans ce plan : l'interface Streamlit, la configuration Streamlit, `src/app.py`, la détection de dérive conceptuelle en production, ainsi que toute décision RH automatisée.
