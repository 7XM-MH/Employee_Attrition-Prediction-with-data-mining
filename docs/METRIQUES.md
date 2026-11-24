# Contrats de métriques — LOT 1

> Ce document clarifie les métriques utilisées dans le projet et réconcilie les deux tables de résultats, afin que l'évaluateur puisse toujours identifier le pipeline, le seuil et le jeu de données associés à chaque nombre.
> Il ne modifie ni le code ML ni les artefacts générés.

Références : [`README.md`](../README.md) (sections 5 et 6), [`results/model_comparison.csv`](../results/model_comparison.csv), [`results/tableau_metriques_rapport.csv`](../results/tableau_metriques_rapport.csv), [`results/cv_summary.csv`](../results/cv_summary.csv).

---

## 1. Notations

- `N` : effectif du jeu de test (split 80/20 stratifié).
- `TP` : vrais positifs (`Yes` prédits et réels), `FP` : faux positifs, `FN` : faux négatifs, `TN` : vrais négatifs.
- `yᵢ` : étiquette réelle codée `No → 0`, `Yes → 1` ; `pᵢ` : probabilité prédite pour la classe `Yes`.
- Prévalence de la classe positive : **9,03 %** sur le jeu complet, soit un taux de baseline AP d'environ **0,0905** sur le test.

---

## 2. Définitions et formules

| Métrique | Formule | Lecture | Point de vigilance |
|---|---|---|---|
| **Accuracy** | `(TP + TN) / N` | Proportion de bonnes prédictions toutes classes confondues | Dominée par les `No` : 0,9095 est atteignable sans détecter aucun départ |
| **Precision** | `TP / (TP + FP)` | Parmi les employés prédits partants, part qui part réellement | Chute quand le seuil est abaissé |
| **Recall** | `TP / (TP + FN)` | Parmi les employés réellement partants, part détectée | Augmente quand le seuil est abaissé |
| **F1** | `2 × Precision × Recall / (Precision + Recall)` = `2TP / (2TP + FP + FN)` | Moyenne harmonique de précision et rappel ; **critère principal du pipeline du guide** | Vaut 0 dès que `TP = 0`, même avec une accuracy élevée |
| **ROC-AUC** | `P(score(positif) > score(négatif))`, aire sous la courbe TPR-FPR | Classement global des scores, indépendant du seuil | Peut paraître correct alors qu'aucun `Yes` n'est prédit au seuil 0,5 |
| **PR-AUC / Average Precision (AP)** | Aire sous la courbe Precision-Recall ; `AP = Σ (Rₙ − Rₙ₋₁) × Pₙ` | Qualité du classement de la classe minoritaire ; **critère principal de l'analyse étendue** | Baseline = prévalence (≈ 0,0905), pas 0,5 |
| **MCC** | `(TP×TN − FP×FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN))` | Corrélation de Matthews, équilibrée même avec classes déséquilibrées | Utile en complément du F1 |
| **Brier** | `(1/N) × Σ (pᵢ − yᵢ)²` | Erreur quadratique des probabilités : calibration | Plus bas est meilleur ; sensible aux probabilités extrêmes |
| **IC 95 % bootstrap** | Tirer `B` échantillons avec remise du test, recalculer la métrique, prendre les percentiles 2,5 et 97,5 | Incertitude empirique de la métrique | Exemple : ROC-AUC de Gradient Boosting `[0,622 ; 0,705]` |
| **Seuil optimal** | `t* = argmax_t F1(t)`, calculé sur l'apprentissage/validation | Seuil de décision appliqué une fois au test | Ici `t* ≈ 0,1176` pour Gradient Boosting ; jamais optimisé sur le test |

---

## 3. Pourquoi F1 et AP avec 9 % de classe positive

La cible `Attrition` est déséquilibrée : environ 10 négatifs pour 1 positif. Un modèle qui prédit toujours `No` obtient une accuracy de **0,9095** tout en ayant un F1 de **0,0000** : il est inutile pour détecter les départs.

- Le **F1** ignore les `TN` et pénalise directement l'absence de `TP` : il mesure la capacité à trouver la classe minoritaire.
- L'**AP / PR-AUC** concentre l'évaluation sur la précision au fil du rappel pour la classe `Yes` ; sa baseline est la prévalence (0,0905), ce qui rend la comparaison significative.
- L'accuracy est conservée comme indication, mais n'est jamais le critère de sélection ni le chiffre mis en avant pour comparer les modèles.

---

## 4. Pourquoi le seuil 0,5 donne un F1 nul

Avec une classe positive rare, les probabilités prédites pour les vrais `Yes` restent le plus souvent inférieures à 0,5. Au seuil par défaut :

1. aucun employé n'est prédit `Yes` → `TP = 0`, `FP = 0`, `FN = 903` environ ;
2. `Precision = 0/0` est ramenée à 0, `Recall = 0`, donc **F1 = 0** ;
3. l'accuracy reste à 0,9095 parce que tous les `No` sont bien classés.

C'est exactement ce que montrent les colonnes `accuracy`, `precision`, `recall`, `f1` de `results/model_comparison.csv` : **0,9095 et 0,0000 pour les neuf modèles**. Le seuil doit donc être optimisé sur l'entraînement/validation, puis appliqué au test. Au seuil `0,1176`, Gradient Boosting passe à `F1 = 0,2527` et `Accuracy = 0,7605` : davantage de vrais départs détectés, au prix de faux positifs assumés.

---

## 5. Pourquoi sélectionner sur la validation croisée, jamais sur le test

Le jeu de test est réservé à l'évaluation finale. Choisir le meilleur modèle d'après le F1 ou l'AP du test reviendrait à utiliser l'information que l'on est censé ne pas connaître et à annoncer une performance optimiste.

- Pipeline du guide (`scripts/train_pipeline.py`) : `StratifiedKFold(n_splits=5)`, sélection sur le **F1 moyen CV** ; `DecisionTree` est retenu (F1 CV `0,1365 ± 0,0230`, F1 test `0,1230`, ROC-AUC `0,5406`).
- Analyse étendue (`scripts/train_models.py`) : `StratifiedKFold(n_splits=5)`, sélection sur l'**AP moyenne CV** ; `Gradient Boosting` est retenu (AP CV `0,1729 ± 0,0136` dans `results/cv_summary.csv`).
- Le test n'est consulté qu'après ces choix, une seule fois, pour produire les métriques rapportées.

---

## 6. Réconciliation des deux tables de métriques

| Source | Réglage | Ce que la colonne représente | Valeurs à citer | Contexte d'usage dans le rapport |
|---|---|---|---|---|
| `results/model_comparison.csv` — colonnes `accuracy`, `precision`, `recall`, `f1` | Seuil fixe **0,5** | Comportement par défaut, avant réglage | Accuracy **0,9095** et F1 **0,0000** pour les 9 modèles | Expliquer le piège du seuil et la baseline ; **ne pas** comparer les modèles avec ces colonnes |
| `results/model_comparison.csv` — colonnes `seuil_opt`, `*_opt`, `roc_auc`, `pr_auc`, `brier`, bornes d'IC | Seuil **optimisé** sur entraînement/validation | Performance opérationnelle après réglage du seuil | Gradient Boosting : seuil **0,1176**, F1 **0,2527**, MCC **0,1630**, ROC-AUC **0,6648**, PR-AUC **0,1732**, Brier **0,0796**, IC ROC-AUC `[0,622 ; 0,705]` | Comparaison étendue, choix du seuil, incertitude |
| `results/tableau_metriques_rapport.csv` | Seuil **optimisé**, table de synthèse du PDF | Mêmes métriques, formatées pour le rapport | Gradient Boosting : Accuracy **0,7605**, F1 **0,2527**, ROC-AUC **0,6648**, PR-AUC **0,1732**, rang moyen **1,0** | Citer dans le rapport **en précisant « au seuil optimisé »** |
| `README.md` §6.1 / `scripts/train_pipeline.py` | Pipeline du **guide**, F1 CV, 6 modèles imposés | Conformité au support de cours | DecisionTree : F1 CV **0,1365 ± 0,0230**, F1 test **0,1230**, ROC-AUC **0,5406** | Citer pour le pipeline imposé et le modèle retenu par le guide |
| `results/cv_summary.csv` | Analyse **étendue**, AP CV, 9 modèles | Critère de sélection du meilleur modèle étendu | Gradient Boosting : AP CV **0,1729 ± 0,0136** ; Arbre de décision : AP CV **0,1456** | Justifier le choix du modèle étendu, jamais le score de test |

**Règle de citation :** toujours indiquer le pipeline (guide ou étendu) et le seuil (0,5 ou optimisé). Ne jamais écrire « accuracy 0,9095 » et « F1 0,2527 » dans la même phrase sans préciser qu'ils proviennent de deux réglages de seuil différents. Le chiffre `0,9095` sert uniquement à illustrer la baseline et le piège du seuil par défaut.
