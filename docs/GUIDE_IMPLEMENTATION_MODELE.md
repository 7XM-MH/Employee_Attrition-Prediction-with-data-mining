# Guide d'implementation du modele de Data Mining

Ce guide est destine aux membres de l'equipe qui entrainent, evaluent ou branchent le modele de classification au site Streamlit. Il decrit le depot tel qu'il existe et les contrats a conserver entre les donnees, le pipeline et l'application. Il s'agit d'un guide de travail : il ne genere ni modele, ni metrique, ni autre artefact.

## 1. Point de depart : depot et donnees

Le projet est un probleme de classification supervisee de l'attrition. Le jeu de donnees present est :

```text
data/raw/Employee_Attrition_DataSet.csv
```

Il contient 10 000 lignes et 14 colonnes. La cible reelle s'appelle `Attrition` et ses valeurs sont `No` et `Yes` (pas `Stayed` et `Left`). Les 12 variables d'entree utilisees par le pipeline sont :

| Type | Colonnes |
|---|---|
| Numeriques (9) | `Age`, `JobLevel`, `YearsAtCompany`, `MonthlyIncome`, `JobSatisfaction`, `WorkLifeBalance`, `DistanceFromHome`, `PerformanceRating`, `TrainingHoursLastYear` |
| Categorielles (3) | `Department`, `OverTime`, `PromotionLast5Years` |
| Cible (a exclure de X) | `Attrition` |
| Identifiant (a exclure de X) | `EmployeeID` |

La classe positive est `Yes`, qui represente environ 9 % des lignes. Le jeu est synthetique, sans valeurs manquantes ni doublons selon la decouverte du projet. Les colonnes `Gender`, `MaritalStatus`, `RemoteWork` et `JobRole` mentionnees dans certains exemples de specification ne sont pas dans ce CSV : ne pas les ajouter au formulaire ou au modele sans changement valide du jeu de donnees et de son contrat.

Le depot contient deja `src/preprocessing.py`, `src/models.py`, `src/evaluation.py`, `scripts/train_pipeline.py` et les cinq pages sous `app/pages/`. Le script d'entrainement est deja implemente ; `models/` et `results/` ne contenaient toutefois pas de modele ni de resultats lors de la verification. Les notebooks presents sont vides. `DISCOVERY.md` contient une synthese utile, mais certaines de ses indications d'etat peuvent devenir obsoletes : verifiez les fichiers actuels avant de vous y fier.

## 2. Contrats d'integration a respecter

Les pages et utilitaires existants attendent les conventions suivantes :

1. **Entrees du modele** : un `pandas.DataFrame` dont les colonnes sont exactement les 12 variables de `ALL_FEATURES` definies dans `src/preprocessing.py` et `app/utils.py`. L'ordre peut etre normalise avant `predict`, mais les noms et types doivent rester coherents.
2. **Cible** : convertir `No` en `0` et `Yes` en `1` pendant l'entrainement. Pour l'affichage, `0` correspond a « Non / reste » et `1` a « Oui / quitte ». Ne pas inclure `Attrition` ni `EmployeeID` dans les donnees envoyees au modele.
3. **Pipeline sauvegarde** : sauvegarder le preprocesseur et le classifieur ensemble dans un seul objet `sklearn.pipeline.Pipeline`. Ainsi, la prediction individuelle et la prediction en lot appellent directement `predict(X)` et `predict_proba(X)` sur les colonnes originales.
4. **Probabilite de depart** : toutes les pages de prediction utilisent la probabilite de la classe positive a l'indice correct (`Yes`/`1`), pas simplement la premiere colonne de `predict_proba`. Verifier `classes_` si l'ordre des classes n'est pas garanti.
5. **Nom du modele par defaut** : l'utilitaire `app.utils.load_model("best_model")` cherche `models/best_model.joblib`. Les modeles individuels sont nommes, par exemple, `models/RandomForest_pipeline.joblib`.
6. **Chemins** : construire les chemins depuis `Path(__file__).resolve()` ou depuis une racine de projet partagee. Ne pas coder de chemin absolu propre a un ordinateur.
7. **Chargement de donnees** : les CSV charges en lot doivent avoir toutes les colonnes de `ALL_FEATURES`. L'identifiant et la cible sont facultatifs dans le fichier de lot, mais ils ne doivent pas etre envoyes au pipeline.

Les definitions des variables figurent a la fois dans `src/preprocessing.py` et `app/utils.py`. Toute modification de schema doit mettre ces definitions et les formulaires/validations des pages a jour ensemble. Eviter de maintenir un troisieme schema independant dans le script d'entrainement.

## 3. Pipeline d'entrainement recommande

Le point d'entree d'entrainement est :

```text
scripts/train_pipeline.py
```

Le pipeline doit rester classique et interpretable au niveau des methodes du cours : Zero-R (`DummyClassifier`), K-NN, arbre de decision, foret aleatoire, Naive Bayes et SVM. N'utiliser ni reseau de neurones, ni LLM comme modele predictif.

### Ordre des operations

1. Verifier que le CSV existe, que les colonnes attendues sont presentes, que `Attrition` contient uniquement `No`/`Yes`, et que les classes ne sont pas vides. Emettre un message explicite en cas de schema invalide.
2. Separer d'abord les donnees en apprentissage et test avec `train_test_split(..., stratify=y, random_state=...)`. Conserver le test hors de tout ajustement, choix d'hyperparametres et choix du modele.
3. Creer un `ColumnTransformer` : mise a l'echelle numerique avec `StandardScaler`, encodage categoriel avec `OneHotEncoder(handle_unknown="ignore")`. Garder ce transformeur dans une `Pipeline` avec le classifieur.
4. Ajuster la recherche d'hyperparametres uniquement sur l'ensemble d'apprentissage avec `StratifiedKFold(n_splits=5, shuffle=True, random_state=...)` et une metrique adaptee a la classe minoritaire (F1 pour `Yes`; ROC-AUC peut etre rapporte en complement).
5. Inclure Zero-R comme baseline. Enregistrer ses resultats de validation croisee aussi, ou indiquer explicitement qu'une valeur CV n'a pas ete calculee ; ne pas la presenter comme un score nul.
6. Selectionner le meilleur modele a partir d'un score de validation croisee calcule sur l'apprentissage (ou d'une regle documentee et definie avant de consulter le test). Evaluer ensuite le modele retenu sur le test. Choisir le gagnant d'apres le F1 du test puis annoncer ce meme test comme evaluation finale introduit un biais de selection.
7. Calculer les metriques du test et sauvegarder le pipeline complet, sans re-entrainer le gagnant sur le test.

Les transformations doivent etre apprises a l'interieur de la pipeline pendant la validation croisee. Ne pas encoder ou normaliser tout le jeu avant le split/CV. Si un reequilibrage (par exemple SMOTE) est requis par l'enseignant, l'appliquer exclusivement dans chaque pli d'entrainement au moyen d'une integration compatible avec `Pipeline` (par exemple `imblearn.pipeline.Pipeline`), jamais avant la separation ou sur le test. Justifier et comparer son effet au baseline.

## 4. Artefacts et formats attendus

Le script actuel vise les fichiers suivants, que les pages lisent sous ces noms :

| Chemin | Utilisation par l'application |
|---|---|
| `models/best_model.joblib` | Modele charge par defaut pour prediction individuelle et par lot |
| `models/ZeroR_pipeline.joblib` et `<NomModele>_pipeline.joblib` | Pipelines sauvegardes par modele, disponibles notamment pour le comparatif |
| `results/metrics.json` | Dictionnaire `{nom_modele: {accuracy, precision, recall, f1, roc_auc}}` pour la page de comparaison |
| `results/roc_data.json` | Dictionnaire `{nom_modele: {fpr, tpr, auc}}` pour tracer les courbes ROC |
| `results/confusion_matrices.json` | Dictionnaire `{nom_modele: [[TN, FP], [FN, TP]]}` avec l'ordre de classes explicite |
| `results/best_hyperparameters.json` | Dictionnaire `{nom_modele: {parametre: valeur}}` |
| `results/cv_results_summary.json` | Dictionnaire `{nom_modele: {mean_cv_f1, std_cv_f1}}` |
| `results/feature_importances.json` | Dictionnaire `{nom_modele: {nom_variable_transformee: importance}}` pour les modeles qui ont une importance native |
| `data/processed/test_set.csv` | Jeu de test conserve pour inspection/reproductibilite ; ne pas le re-utiliser pour ajuster le modele |

Conserver les cles de modele (`ZeroR`, `KNN`, `DecisionTree`, `RandomForest`, `NaiveBayes`, `SVM`) et les noms de champs ci-dessus : `app/pages/3_Performance_Modeles.py` les utilise directement. Les pages de facteurs consultent les importances et le modele sauvegarde. Si un nouveau fichier de metadonnees est ajoute, documenter son schema et son consommateur dans l'application.

La page d'accueil peut avoir besoin du nom et du critere de selection du meilleur modele ; ne pas l'inferer a partir d'une metrique de test si le choix a ete fait sur la CV. Enregistrer cette information dans un petit artefact explicite, ou la determiner avec la meme regle CV documentee. Les nombres NumPy doivent etre convertis en types Python natifs avant serialisation JSON.

## 5. Connexion aux pages Streamlit

- **Accueil (`app/Home.py`)** : charger les metriques disponibles pour les indicateurs. Si le modele ou ses resultats manquent, afficher l'avertissement avec la commande de generation exacte (`python scripts/train_pipeline.py`), sans erreur non geree.
- **Dashboard (`app/pages/1_Dashboard.py`)** : lire les colonnes depuis le dataset reel ; les distributions et filtres doivent utiliser les modalites presentes dans les donnees, pas des variables hypothetique de l'exemple.
- **Prediction individuelle (`app/pages/2_Prediction.py`)** : construire un DataFrame d'une ligne, avec les noms des 12 features et des types compatibles ; afficher prediction, probabilite de `Yes` et seuil de risque clairement. Le modele charge doit rester le pipeline complet.
- **Performance (`app/pages/3_Performance_Modeles.py`)** : lire les JSON de `results/` selon le contrat ci-dessus. Les metriques de test et les scores de CV doivent etre nommes separement.
- **Facteurs (`app/pages/4_Facteurs_Demission.py`)** : les importances natives existent pour les arbres/foret. Pour K-NN, SVM ou Naive Bayes, employer une methode explicable compatible, comme une importance par permutation sur un echantillon, ou indiquer clairement que la mesure n'est pas disponible. SHAP est une aide d'explicabilite, pas un modele predictif.
- **Prediction en lot (`app/pages/5_Prediction_en_lot.py`)** : valider toutes les colonnes avant prediction, signaler celles manquantes, ignorer ou retirer explicitement les colonnes supplementaires, et aligner le DataFrame sur `ALL_FEATURES`. Trier les resultats selon la probabilite de classe `Yes`.

Le pied de page et les resultats doivent rappeler que les donnees sont synthetiques et que le modele ne convient pas aux decisions reelles de ressources humaines. Les scores ne prouvent pas la causalite et ne doivent pas etre interpretes comme une evaluation individuelle fiable.

## 6. Procedure d'equipe

1. **Confirmer le schema** : consulter `DISCOVERY.md` puis verifier le CSV et `src/preprocessing.py`. En cas de divergence, le CSV est la source de verite, et le schema doit etre mis a jour partout.
2. **Stabiliser le contrat** : convenir des 12 features, du codage `No=0`/`Yes=1`, de la classe positive, des noms de fichiers et des cles JSON avant de travailler en parallele.
3. **Entrainement et evaluation** : maintenir le pipeline et le split dans `scripts/train_pipeline.py` ; separer selection CV et evaluation finale sur test.
4. **Generer les artefacts** depuis la racine du depot :

   ```bash
   python scripts/train_pipeline.py
   ```

   Sous Windows, utiliser l'environnement du projet, par exemple :

   ```powershell
   .\.venv\Scripts\python.exe .\scripts\train_pipeline.py
   ```

5. **Verifier le resultat** : confirmer que `best_model.joblib`, les six pipelines et les JSON attendus existent, qu'ils se chargent, que leurs cles sont coherentes, et que `predict_proba` renvoie une probabilite finie pour `Yes`.
6. **Tester l'integration** : lancer depuis la racine :

   ```bash
   streamlit run app/Home.py
   pytest
   ```

   Verifier manuellement une prediction individuelle, le comparatif des six modeles, les graphiques de facteurs et un CSV de lot contenant les 12 colonnes. Tester aussi le comportement lorsque les artefacts sont absents.
7. **Mettre a jour la documentation** si les noms de colonnes, l'algorithme de selection, les metriques ou la structure d'artefacts changent.

Les dependances du projet sont declarees dans `requirements.txt`. Installer ou mettre a jour une dependance dans l'environnement virtuel du projet uniquement si necessaire ; garder les versions compatibles avec Python 3.10+ et Streamlit Community Cloud.

## 7. Points de vigilance releves dans le depot

- Plusieurs avertissements de pages pointent encore vers `python src/train.py`, qui n'est pas le script d'entrainement de ce depot. La commande actuelle est `python scripts/train_pipeline.py`; corriger les messages UI lorsqu'on travaille sur ces pages.
- `src/app.py` est un lanceur qui demarre `app/Home.py` via Streamlit. Pour respecter le lancement local exige par le cours, tester directement `streamlit run app/Home.py`.
- Les references a `train.csv`/`test.csv` dans des instructions generiques ne correspondent pas aux fichiers presents. Le script cree actuellement `data/processed/test_set.csv` a partir du split du CSV source.
- La liste de features du modele n'inclut ni `EmployeeID` ni `Attrition`. La prediction par lot doit suivre la meme regle que la prediction individuelle.
- Ne jamais annoncer une metrique ou une fonctionnalite comme validee sans l'avoir executee dans l'environnement du projet.
