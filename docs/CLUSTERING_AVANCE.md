# Clustering avancé — LOT 3 (chapitres 7 à 10 du cours)

> Pipeline non supervisé complémentaire aux quatre méthodes historiques. Ce document
> enregistre le choix de `k`, les sept méthodes comparées, les indices de validation
> et les limites d'interprétation. Aucune prédiction n'est destinée à une décision RH réelle.

Sources : [`src/unsupervised.py`](../src/unsupervised.py),
[`scripts/run_unsupervised_advanced.py`](../scripts/run_unsupervised_advanced.py),
[`results/clustering_advanced.csv`](../results/clustering_advanced.csv),
[`tests/test_unsupervised_advanced.py`](../tests/test_unsupervised_advanced.py).

---

## 1. Données utilisées et absence de fuite

La matrice est produite par `encode_features()` : 10 000 lignes × 20 colonnes encodées,
à partir des 12 variables de `FEATURE_COLS` (5 quantitatives standardisées, 4 ordinales,
2 binaires, 1 nominale en one-hot). **`EmployeeID` et `Attrition` sont exclus avant tout
ajustement** ; ils n'apparaissent ni dans la matrice, ni dans le choix de `k`, ni dans les
médoïdes. Le taux d'attrition par cluster est calculé *après coup*, uniquement pour décrire
les groupes (colonne `attrition_rate_by_cluster` du CSV).

Le test `test_no_target_leakage_from_extra_column` modifie une colonne `cible`, la cible
`Attrition` et l'identifiant : la matrice et les étiquettes PAM restent bit à bit identiques.

---

## 2. Choix du nombre de groupes

`choose_k(matrix, 2, 10)` ajuste K-Means avec la graine 42 pour chaque `k`, puis croise
deux critères :

| k | Inertie | Silhouette | Davies-Bouldin | Calinski-Harabasz |
|---:|---:|---:|---:|---:|
| 2 | 82 068,99 | **0,1875** | 1,8683 | 2 681,2074 |
| **3** | **76 182,98** | **0,1083** | 2,4840 | 1 830,2270 |
| 4 | 72 372,70 | 0,1051 | 2,3043 | 1 459,6855 |
| 5 | 69 239,49 | 0,0990 | 2,2848 | 1 257,2641 |
| 6 | 66 507,07 | 0,0895 | 2,3637 | 1 129,1488 |
| 7 | 64 500,64 | 0,0858 | 2,2689 | 1 021,9424 |
| 8 | 62 753,82 | 0,0843 | 2,1975 | 939,9753 |
| 9 | 61 257,66 | 0,0812 | 2,1597 | 872,9851 |
| 10 | 59 924,98 | 0,0798 | 2,2694 | 817,8508 |

**k retenu : 3.** La meilleure silhouette brute est observée pour `k = 2` (0,1875), mais
la plus forte baisse relative d'inertie se produit pour `k = 3` :
`(82 068,99 − 76 182,98) / 82 068,99 = 7,17 %`. Le coude est donc en `k = 3`. La règle
retenue est : *choisir la meilleure silhouette, sauf si elle se situe avant le coude
d'inertie, auquel cas on garde le coude*. Cela évite de figer une partition trop grossière
qui « gagne » seulement parce que deux groupes sont trivialement séparables. Le `k = 4`
des lots précédents reste dans le CSV pour continuité, mais n'est plus le choix par défaut.

Voir [`37_choix_k.png`](../reports/figures/37_choix_k.png).

---

## 3. Formules des indices

Soit `C_i` un cluster, `c_i` son centroïde, `n` le nombre de points.

- **Silhouette** (échantillon déterministe de 2 000 lignes, graine 42) : pour un point
  `x ∈ C_i`, `a(x)` est la distance moyenne aux autres points de `C_i` et `b(x)` la plus
  petite distance moyenne à un autre cluster. `s(x) = (b(x) − a(x)) / max(a(x), b(x))`.
  La silhouette globale est la moyenne des `s(x)`, dans `[-1, 1]` ; **plus elle est élevée,
  mieux c'est**. Le bruit DBSCAN (`-1`) est exclu.
- **Davies-Bouldin** (`sklearn.metrics.davies_bouldin_score`) :
  `DB = (1/k) Σ_i max_{j≠i} (S_i + S_j) / d(c_i, c_j)`, où `S_i` est la dispersion moyenne
  du cluster `i` par rapport à son centroïde. Minorant théorique 0 ; **plus c'est faible,
  mieux c'est**.
- **Dunn** (implémentation locale, échantillon déterministe de 2 000 lignes) :
  `Dunn = min_{i≠j} min_{x∈C_i, y∈C_j} d(x,y) / max_l diam(C_l)`. **Plus c'est élevé, mieux
  c'est**. Le diamètre est la distance maximale intra-cluster.
- **Calinski-Harabasz** est également reporté pour le balayage de `k` (variance inter /
  variance intra, plus c'est élevé mieux c'est).

Comparaison visuelle : [`40_indices_clustering.png`](../reports/figures/40_indices_clustering.png).

---

## 4. Tableau comparatif des sept méthodes

Valeurs obtenues avec `k = 3` sauf PCA (2 composantes) et DBSCAN (`eps = 0,18`,
`min_samples = 25`). Les indices de DBSCAN sont calculés en excluant le bruit.

| Méthode | Principe | Silhouette | Davies-Bouldin | Dunn | Tailles des groupes | Bruit |
|---|---|---:|---:|---:|---|---:|
| PCA | Projection 2D, 40,58 % de variance expliquée | — | — | — | pas de partition | — |
| KMeans | Partition euclidienne, `n_init=10`, graine 42 | 0,1083 | 2,4840 | 0,0702 | 2 201 / 3 819 / 3 980 | 0 |
| AgglomerativeClustering | Ward + connectivité 10-PPV | 0,0847 | 2,6413 | 0,0849 | 3 969 / 4 245 / 1 786 | 0 |
| DBSCAN | Densité sur la projection PCA 2D | 0,1756 | 1,6513 | 0,2103 | 8 566 / 40 | 1 394 |
| KMedoids | PAM simplifié, médoïdes = lignes réelles | 0,0890 | 2,5780 | 0,0663 | 2 634 / 4 475 / 2 891 | 0 |
| DIANA | Divisif : scission du plus grand diamètre | 0,0347 | 4,0884 | 0,0464 | 4 900 / 2 100 / 3 000 | 0 |
| CosineKMeans | K-Means sur matrice L2-normalisée | 0,1109 | 2,5109 | 0,0552 | 3 243 / 3 963 / 2 794 | 0 |

Lectures : DBSCAN affiche les meilleurs indices mais laisse 1 394 points de bruit et un
groupe de 40 employés seulement ; son score n'est donc pas comparable à une partition
couvrant tout le jeu. CosineKMeans et KMeans sont très proches, ce qui indique que la
normalisation L2 n'apporte pas de séparation supplémentaire ici. DIANA est nettement moins
bon : la règle de scission (30 % des points les plus éloignés de la graine) produit des
groupes déséquilibrés sur ce jeu synthétique.

Projections : [`39_kmedoids_diana.png`](../reports/figures/39_kmedoids_diana.png) et
[`41_clusters_cosinus.png`](../reports/figures/41_clusters_cosinus.png).

---

## 5. Interprétation a posteriori des groupes

Les taux ci-dessous sont **descriptifs** : `Attrition` n'a servi ni à l'encodage, ni au
choix de `k`, ni à l'affectation des points. Pour KMeans à `k = 3` :

| Cluster | Taille | Taux de départ observé |
|---:|---:|---:|
| 0 | 2 201 | 5,77 % |
| 1 | 3 819 | 10,76 % |
| 2 | 3 980 | 9,17 % |

L'écart entre le groupe le plus à risque (10,76 %) et le plus faible (5,77 %) est d'environ
5 points, mais la silhouette globale reste faible (0,1083) : la segmentation est donc
**explorable mais non décisionnelle**. Les taux des autres méthodes sont dans
[`results/clustering_advanced.csv`](../results/clustering_advanced.csv).

---

## 6. Limite du dendrogramme sur 10 000 lignes

Une hiérarchie complète demanderait une matrice de distances de 10 000 × 10 000.
[`38_dendrogramme.png`](../reports/figures/38_dendrogramme.png) est donc calculé avec
`scipy.cluster.hierarchy.linkage(..., method="ward")` sur un **échantillon déterministe de
2 000 lignes** (graine 42), puis tronqué aux 30 dernières fusions avec `dendrogram(...,
truncate_mode="lastp", p=30)`. La figure illustre la structure hiérarchique ; elle ne sert
pas à fixer `k` et ne doit pas être lue comme une partition des 10 000 employés.

---

## 7. État de compatibilité et correctif prévu au lot 4

Le refactoring de `src/preprocessing.py` a renommé `ALL_FEATURES` en `FEATURE_COLS`,
`load_raw_data()` en `load_raw(path)` et supprimé `DATA_PATH`. Les références dans
`src/unsupervised.py` ont été corrigées dans ce lot, avec un test de non-régression
(`test_import_unsupervised`) qui vérifie que `import src.unsupervised` réussit.

**`scripts/train_pipeline.py` importe encore `ALL_FEATURES`, `DATA_PATH` et `prepare_data`.
Sa commande de régénération supervisée ne fonctionne donc pas aujourd'hui. Le correctif de
compatibilité est prévu au lot 4**, sans réécriture du pipeline dans ce lot. Les commandes
de clustering avancé, elles, sont exécutables depuis la racine :

```bash
python scripts/run_unsupervised_advanced.py
python -m pytest tests -q
```
