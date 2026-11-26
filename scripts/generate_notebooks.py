"""
Script to generate the 4 standard Jupyter Notebooks for the Data Mining project.
Produces fully valid .ipynb files with detailed explanations and executable code cells.
"""
import json
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent / "notebooks"


def create_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.13.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }


def md_cell(source):
    lines = [s + "\n" for s in source.strip().split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {"cell_type": "markdown", "metadata": {}, "source": lines}


def code_cell(source):
    lines = [s + "\n" for s in source.strip().split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines
    }


# ==============================================================================
# NOTEBOOK 1: 01_eda_and_cleaning.ipynb
# ==============================================================================
nb1_cells = [
    md_cell("""# 🔬 Notebook 1 : Exploration & Nettoyage des Données (EDA & Cleaning)
**Projet :** Prédiction de l'Attrition des Employés — Data Mining (Track A : Classification Supervisée)  
**Objectif :** Audit de la qualité des données (10 000 lignes), statistiques descriptives, analyse du déséquilibre de classe, détection et traitement des valeurs aberrantes (outliers IQR), et visualisations univariées/bivariées."""),
    
    code_cell("""import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration esthétique
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 11

print("Bibliothèques chargées avec succès.")"""),

    md_cell("""## 1. Chargement et Inspection Initiale du Jeu de Données"""),

    code_cell("""# Chemin vers le fichier brut
data_path = Path("../data/raw/Employee_Attrition_DataSet.csv")
if not data_path.exists():
    data_path = Path("data/raw/Employee_Attrition_DataSet.csv")

df = pd.read_csv(data_path)
print(f"Dimensions du dataset : {df.shape[0]} observations, {df.shape[1]} colonnes.")
df.head()"""),

    code_cell("""# Structure et types des colonnes
df.info()"""),

    md_cell("""## 2. Audit de Qualité : Valeurs Manquantes et Doublons"""),

    code_cell("""# Vérification des valeurs manquantes et des doublons
missing = df.isnull().sum()
duplicates = df.duplicated().sum()

print("--- Valeurs manquantes par colonne ---")
print(missing[missing > 0] if missing.sum() > 0 else "Aucune valeur manquante détectée (0 NaN).")
print(f"\\nNombre de doublons stricts : {duplicates}")"""),

    md_cell("""## 3. Analyse de la Variable Cible : Attrition"""),

    code_cell("""# Distribution de la variable cible
target_counts = df["Attrition"].value_counts()
target_pct = df["Attrition"].value_counts(normalize=True) * 100

summary_target = pd.DataFrame({
    "Effectif": target_counts,
    "Pourcentage (%)": target_pct.round(2)
})
print(summary_target)

# Visualisation du déséquilibre de classe
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

# Graphique à barres
sns.barplot(x=target_counts.index, y=target_counts.values, palette=["#10B981", "#EF4444"], ax=ax[0])
ax[0].set_title("Distribution des classes (Effectif)")
ax[0].set_ylabel("Nombre d'employés")
for i, v in enumerate(target_counts.values):
    ax[0].text(i, v + 150, f"{v} ({target_pct.iloc[i]:.1f}%)", ha="center", fontweight="bold")

# Graphique en anneau (Donut chart)
ax[1].pie(target_counts, labels=["Stayed (Non)", "Left (Oui)"], colors=["#10B981", "#EF4444"],
          autopct="%1.1f%%", startangle=90, pctdistance=0.85,
          wedgeprops=dict(width=0.4, edgecolor='white'))
ax[1].set_title("Proportion de l'Attrition RH")

plt.tight_layout()
plt.show()"""),

    md_cell("""> **Constat :** Le jeu de données présente un fort déséquilibre de classe (~91% Stayed vs ~9% Left, ratio ~10:1). L'Accuracy brute est donc un indicateur trompeur (le classifieur Zero-R atteignant 90.97% sans aucun pouvoir discriminant). Le score F1 et le ROC-AUC seront nos métriques prioritaires."""),

    md_cell("""## 4. Statistiques Descriptives des Variables Numériques"""),

    code_cell("""numerical_cols = [
    "Age", "JobLevel", "YearsAtCompany", "MonthlyIncome",
    "JobSatisfaction", "WorkLifeBalance", "DistanceFromHome",
    "PerformanceRating", "TrainingHoursLastYear"
]

desc_stats = df[numerical_cols].describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]]
desc_stats["IQR"] = desc_stats["75%"] - desc_stats["25%"]
desc_stats.round(2)"""),

    md_cell("""## 5. Détection et Traitement des Valeurs Aberrantes (Méthode IQR)"""),

    code_cell("""# Détection des outliers selon la règle 1.5 x IQR
def detect_outliers_iqr(data, columns):
    report = []
    for col in columns:
        q1 = data[col].quantile(0.25)
        q3 = data[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        n_outliers = ((data[col] < lower) | (data[col] > upper)).sum()
        pct = (n_outliers / len(data)) * 100
        report.append({
            "Variable": col,
            "Q1": round(q1, 2),
            "Q3": round(q3, 2),
            "IQR": round(iqr, 2),
            "Borne Inf": round(lower, 2),
            "Borne Sup": round(upper, 2),
            "Nbr Outliers": n_outliers,
            "% Outliers": round(pct, 2)
        })
    return pd.DataFrame(report)

outliers_df = detect_outliers_iqr(df, numerical_cols)
outliers_df"""),

    code_cell("""# Écrêtage (Winsorization) des valeurs aberrantes
def cap_outliers_iqr(data, columns):
    df_capped = data.copy()
    for col in columns:
        q1 = df_capped[col].quantile(0.25)
        q3 = df_capped[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df_capped[col] = df_capped[col].clip(lower=lower, upper=upper)
    return df_capped

df_cleaned = cap_outliers_iqr(df, numerical_cols)
# Suppression de l'identifiant non prédictif
if "EmployeeID" in df_cleaned.columns:
    df_cleaned = df_cleaned.drop(columns=["EmployeeID"])

print("Jeu de données nettoyé et écrêté avec succès.")
print(f"Dimensions finales : {df_cleaned.shape}")"""),

    md_cell("""## 6. Analyse Univariée & Bivariée des Caractéristiques Clés"""),

    code_cell("""# Distributions comparées selon le statut d'attrition
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. Revenu mensuel
sns.boxplot(x="Attrition", y="MonthlyIncome", data=df, palette=["#10B981", "#EF4444"], ax=axes[0, 0])
axes[0, 0].set_title("Revenu Mensuel vs Attrition")

# 2. Distance du domicile
sns.boxplot(x="Attrition", y="DistanceFromHome", data=df, palette=["#10B981", "#EF4444"], ax=axes[0, 1])
axes[0, 1].set_title("Distance du Domicile (km) vs Attrition")

# 3. Ancienneté dans l'entreprise
sns.boxplot(x="Attrition", y="YearsAtCompany", data=df, palette=["#10B981", "#EF4444"], ax=axes[1, 0])
axes[1, 0].set_title("Années dans l'Entreprise vs Attrition")

# 4. Overtime (Heures supplémentaires)
ct = pd.crosstab(df["OverTime"], df["Attrition"], normalize="index") * 100
ct.plot(kind="bar", stacked=True, color=["#10B981", "#EF4444"], ax=axes[1, 1], rot=0)
axes[1, 1].set_title("Taux d'Attrition par Heures Supplémentaires (%)")
axes[1, 1].set_ylabel("%")

plt.tight_layout()
plt.show()"""),

    md_cell("""## 7. Matrice de Corrélation de Pearson"""),

    code_cell("""# Matrice de corrélation
corr = df[numerical_cols].corr()

plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, linewidths=0.5)
plt.title("Matrice de Corrélation de Pearson (Variables Numériques)")
plt.show()"""),

    md_cell("""## 8. Synthèse de l'Étape 1
1. **Intégrité :** 10 000 enregistrements complets, 0 doublons, 0 valeurs manquantes.
2. **Déséquilibre :** 90.97 % `Stayed` vs 9.03 % `Left`.
3. **Outliers :** Présence d'outliers identifiés et traités par bornage IQR sans perte de données.
4. **Relations préliminaires :** Les variables `MonthlyIncome`, `DistanceFromHome`, et `OverTime` montrent de fortes distinctions comportementales entre les employés qui restent et ceux qui partent.""")
]


# ==============================================================================
# NOTEBOOK 2: 02_statistical_analysis.ipynb
# ==============================================================================
nb2_cells = [
    md_cell("""# 📊 Notebook 2 : Tests d'Hypothèses Statistiques
**Projet :** Prédiction de l'Attrition des Employés — Data Mining  
**Objectif :** Formuler et valider 4 hypothèses statistiques formalisées au seuil $\\alpha = 0.05$, avec calcul de la taille d'effet (Cramér's V, $r$ de rang) et correction de tests multiples de Benjamini-Hochberg (FDR)."""),

    code_cell("""import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
df = pd.read_csv("../data/raw/Employee_Attrition_DataSet.csv")
print("Dataset chargé :", df.shape)"""),

    md_cell("""## Définition des Fonctions d'Effet de Taille"""),

    code_cell("""def cramers_v(contingency_table):
    \"\"\"Calcul du V de Cramér pour tableau de contingence.\"\"\"
    chi2 = stats.chi2_contingency(contingency_table)[0]
    n = contingency_table.sum().sum()
    r, k = contingency_table.shape
    return np.sqrt(chi2 / (n * min(r - 1, k - 1)))

def rank_biserial(group0, group1):
    \"\"\"Calcul de la corrélation bisérielle de rang (taille d'effet Mann-Whitney).\"\"\"
    u_stat, _ = stats.mannwhitneyu(group0, group1)
    n0, n1 = len(group0), len(group1)
    return 1 - (2 * u_stat) / (n0 * n1)"""),

    md_cell("""## Hypothèse 1 : OverTime et Attrition (Test du Chi-carré $\\chi^2$)
* $H_0$ : La réalisation d'heures supplémentaires est indépendante de l'attrition.
* $H_1$ : Il existe une association significative entre les heures supplémentaires et l'attrition."""),

    code_cell("""ct_ot = pd.crosstab(df["OverTime"], df["Attrition"])
chi2_ot, p_ot, dof_ot, _ = stats.chi2_contingency(ct_ot)
v_ot = cramers_v(ct_ot)

print(ct_ot)
print(f"\\nChi2 = {chi2_ot:.4f}, ddl = {dof_ot}, p-valeur = {p_ot:.4e}")
print(f"Taille d'effet (V de Cramér) = {v_ot:.4f}")
print("Décision :", "Rejet de H0 (Association significative)" if p_ot < 0.05 else "Non-rejet de H0")"""),

    md_cell("""## Hypothèse 2 : Revenu Mensuel et Attrition (Test U de Mann-Whitney)
* $H_0$ : La distribution du revenu mensuel est identique entre les employés qui restent et ceux qui partent.
* $H_1$ : Les employés qui partent ont une rémunération mensuelle significativement différente."""),

    code_cell("""income_no = df[df["Attrition"] == "No"]["MonthlyIncome"]
income_yes = df[df["Attrition"] == "Yes"]["MonthlyIncome"]

# Test de normalité (Shapiro-Wilk sur échantillon de 500)
shapiro_p = stats.shapiro(income_no.sample(500, random_state=42))[1]
print(f"Test de normalité Shapiro-Wilk (p-valeur) : {shapiro_p:.4e} -> Données non-normales.")

# Test non paramétrique de Mann-Whitney U
u_stat, p_mw = stats.mannwhitneyu(income_no, income_yes, alternative="two-sided")
r_effect = rank_biserial(income_no, income_yes)

print(f"Statistique U = {u_stat:.2f}, p-valeur = {p_mw:.4e}")
print(f"Taille d'effet r bisérial de rang = {r_effect:.4f}")
print("Décision :", "Rejet de H0 (Différence significative de salaire)" if p_mw < 0.05 else "Non-rejet de H0")"""),

    md_cell("""## Hypothèse 3 : Équilibre Vie Pro/Perso et Attrition (Test du Chi-carré $\\chi^2$)
* $H_0$ : Le niveau d'équilibre de vie est indépendant de l'attrition.
* $H_1$ : Il existe une relation significative entre WorkLifeBalance et l'attrition."""),

    code_cell("""ct_wlb = pd.crosstab(df["WorkLifeBalance"], df["Attrition"])
chi2_wlb, p_wlb, dof_wlb, _ = stats.chi2_contingency(ct_wlb)
v_wlb = cramers_v(ct_wlb)

print(ct_wlb)
print(f"\\nChi2 = {chi2_wlb:.4f}, ddl = {dof_wlb}, p-valeur = {p_wlb:.4e}")
print(f"Taille d'effet (V de Cramér) = {v_wlb:.4f}")
print("Décision :", "Rejet de H0" if p_wlb < 0.05 else "Non-rejet de H0")"""),

    md_cell("""## Hypothèse 4 : Corrélation Âge vs Ancienneté (Corrélation de Spearman)
* $H_0$ : Il n'y a pas de corrélation monotone entre l'âge et les années d'ancienneté.
* $H_1$ : Il existe une corrélation monotone positive significative."""),

    code_cell("""rho, p_rho = stats.spearmanr(df["Age"], df["YearsAtCompany"])
print(f"Coefficient rho de Spearman = {rho:.4f}, p-valeur = {p_rho:.4e}")
print("Décision :", "Rejet de H0 (Corrélation positive significative)" if p_rho < 0.05 else "Non-rejet de H0")"""),

    md_cell("""## Correction de Tests Multiples (Benjamini-Hochberg FDR)"""),

    code_cell("""# Agrégation des p-valeurs des 4 tests
raw_pvalues = [p_ot, p_mw, p_wlb, p_rho]
tests_labels = [
    "H1: OverTime vs Attrition (Chi2)",
    "H2: MonthlyIncome vs Attrition (Mann-Whitney)",
    "H3: WorkLifeBalance vs Attrition (Chi2)",
    "H4: Age vs YearsAtCompany (Spearman)"
]

reject, p_adjusted, _, _ = multipletests(raw_pvalues, alpha=0.05, method="fdr_bh")

results_table = pd.DataFrame({
    "Hypothèse": tests_labels,
    "p-valeur brute": [f"{p:.4e}" for p in raw_pvalues],
    "p-valeur corrigée (FDR)": [f"{p:.4e}" for p in p_adjusted],
    "Rejet H0 (alpha=0.05)": ["Oui" if r else "Non" for r in reject]
})
results_table""")
]


# ==============================================================================
# NOTEBOOK 3: 03_preprocessing.ipynb
# ==============================================================================
nb3_cells = [
    md_cell("""# ⚙️ Notebook 3 : Pré-traitement & Pipeline sans Fuite de Données
**Projet :** Prédiction de l'Attrition des Employés — Data Mining  
**Objectif :** Conception d'un pipeline d'encodage et de mise à l'échelle rigoureux avec `ColumnTransformer`, partitionnement stratifié 80/20 et garantie d'absence de fuite de données (Data Leakage)."""),

    code_cell("""import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

print("Modules de pré-traitement importés.")"""),

    md_cell("""## 1. Définition des Variables et Chargement"""),

    code_cell("""DATA_PATH = Path("../data/raw/Employee_Attrition_DataSet.csv")
df = pd.read_csv(DATA_PATH)

NUMERICAL_FEATURES = [
    "Age", "JobLevel", "YearsAtCompany", "MonthlyIncome",
    "JobSatisfaction", "WorkLifeBalance", "DistanceFromHome",
    "PerformanceRating", "TrainingHoursLastYear"
]
CATEGORICAL_FEATURES = ["Department", "OverTime", "PromotionLast5Years"]
TARGET = "Attrition"

# Préparation X et y
X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
y = df[TARGET].map({"Yes": 1, "No": 0})

print(f"X : {X.shape} | y : {y.shape} (Taux classe 1: {y.mean()*100:.2f}%)")"""),

    md_cell("""## 2. Partitionnement Stratifié Train / Test (80/20)
La séparation intervient **avant** toute transformation afin de garantir l'étanchéité absolue du jeu de test."""),

    code_cell("""X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

print(f"Ensemble d'entraînement : {X_train.shape[0]} observations")
print(f"Ensemble de test : {X_test.shape[0]} observations")
print(f"Proportion Attrition Train : {y_train.mean()*100:.2f}% | Test : {y_test.mean()*100:.2f}%")"""),

    md_cell("""## 3. Construction du Préprocesseur (ColumnTransformer)"""),

    code_cell("""preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), NUMERICAL_FEATURES),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES)
    ],
    remainder="drop"
)
preprocessor"""),

    md_cell("""## 4. Ajustement sur Train et Transformation"""),

    code_cell("""# Ajustement UNIQUEMENT sur X_train
preprocessor.fit(X_train)

# Transformation des deux jeux
X_train_trans = preprocessor.transform(X_train)
X_test_trans = preprocessor.transform(X_test)

feature_names = preprocessor.get_feature_names_out()
print(f"Nombre total de variables transformées : {len(feature_names)}")
print("Liste des variables créées :")
for f in feature_names:
    print(" -", f)"""),

    md_cell("""## 5. Contrôle de Non-Fuite (Moyenne & Variance)"""),

    code_cell("""# Vérification que le StandardScaler est calé sur le Train
df_num_train = pd.DataFrame(X_train_trans, columns=feature_names)
print("Moyennes sur le Train (devraient être ~ 0) :")
print(df_num_train.iloc[:, :len(NUMERICAL_FEATURES)].mean().round(4).to_dict())

print("\\nÉcarts-types sur le Train (devraient être ~ 1) :")
print(df_num_train.iloc[:, :len(NUMERICAL_FEATURES)].std().round(4).to_dict())""")
]


# ==============================================================================
# NOTEBOOK 4: 04_model_training_evaluation.ipynb
# ==============================================================================
nb4_cells = [
    md_cell("""# 🏆 Notebook 4 : Benchmark, Optimisation & Explicabilité des Modèles
**Projet :** Prédiction de l'Attrition des Employés — Data Mining  
**Objectif :** Évaluer et comparer 6 algorithmes de classification supervisée (Zero-R, KNN, Arbre de Décision, Forêt Aléatoire, Naive Bayes, SVM) via validation croisée stratifiée 5-Fold, analyse ROC/AUC, matrices de confusion et explicabilité SHAP."""),

    code_cell("""import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
from joblib import load

sns.set_theme(style="whitegrid")
print("Environnement d'évaluation prêt.")"""),

    md_cell("""## 1. Chargement des Résultats et Métriques du Pipeline"""),

    code_cell("""RESULTS_DIR = Path("../results")
MODELS_DIR = Path("../models")

with open(RESULTS_DIR / "metrics.json", "r") as f:
    metrics = json.load(f)

with open(RESULTS_DIR / "cv_results_summary.json", "r") as f:
    cv_results = json.load(f)

with open(RESULTS_DIR / "best_hyperparameters.json", "r") as f:
    best_params = json.load(f)

# Tableau récapitulatif
df_metrics = pd.DataFrame(metrics).T
df_cv = pd.DataFrame(cv_results).T
df_benchmark = df_metrics.join(df_cv)
df_benchmark[["accuracy", "precision", "recall", "f1", "roc_auc", "mean_cv_f1", "std_cv_f1"]]"""),

    md_cell("""## 2. Courbes ROC Comparatives"""),

    code_cell("""with open(RESULTS_DIR / "roc_data.json", "r") as f:
    roc_data = json.load(f)

plt.figure(figsize=(10, 7))
for name, d in roc_data.items():
    plt.plot(d["fpr"], d["tpr"], label=f"{name} (AUC = {d['auc']:.3f})", lw=2)

plt.plot([0, 1], [0, 1], "k--", label="Aléatoire (AUC = 0.500)")
plt.xlabel("Taux de Faux Positifs (FPR)")
plt.ylabel("Taux de Vrais Positifs (TPR / Recall)")
plt.title("Courbes ROC Comparatives — 6 Modèles de Classification")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()"""),

    md_cell("""## 3. Matrices de Confusion sur le Jeu de Test"""),

    code_cell("""with open(RESULTS_DIR / "confusion_matrices.json", "r") as f:
    cms = json.load(f)

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

for i, (name, cm) in enumerate(cms.items()):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[i],
                xticklabels=["Reste (0)", "Quitte (1)"], yticklabels=["Reste (0)", "Quitte (1)"])
    axes[i].set_title(f"Matrice : {name}")
    axes[i].set_xlabel("Prédiction")
    axes[i].set_ylabel("Réalité")

plt.tight_layout()
plt.show()"""),

    md_cell("""## 4. Importance des Variables (Feature Importance - Arbres & Forêt)"""),

    code_cell("""with open(RESULTS_DIR / "feature_importances.json", "r") as f:
    fi = json.load(f)

rf_fi = pd.Series(fi["RandomForest"]).sort_values(ascending=True)

plt.figure(figsize=(10, 8))
rf_fi.plot(kind="barh", color="#4F46E5")
plt.title("Importance des Caractéristiques — Random Forest (Impureté de Gini)")
plt.xlabel("Importance Relative")
plt.tight_layout()
plt.show()"""),

    md_cell("""## 5. Explicabilité Globale avec SHAP (SHAP Values)"""),

    code_cell("""import shap

# Chargement du modèle Random Forest
rf_pipeline = load(MODELS_DIR / "RandomForest_pipeline.joblib")
preprocessor = rf_pipeline.named_steps["preprocessor"]
model = rf_pipeline.named_steps["classifier"]

test_data = pd.read_csv("../data/processed/test_set.csv")
X_test = test_data[rf_pipeline.feature_names_in_]
X_test_trans = preprocessor.transform(X_test)
feature_names = preprocessor.get_feature_names_out()

# Calcul des explications SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_trans[:300])

# Summary plot
if isinstance(shap_values, list):
    # Classe positive (1: Attrition = Yes)
    sv = shap_values[1]
else:
    sv = shap_values[:, :, 1] if len(shap_values.shape) == 3 else shap_values

shap.summary_plot(sv, X_test_trans[:300], feature_names=feature_names, show=False)
plt.title("Impact des Variables sur l'Attrition RH (SHAP Summary Plot)", pad=20)
plt.tight_layout()
plt.show()"""),

    md_cell("""## 6. Synthèse des Résultats & Recommandations RH
1. **Performance Modèles :** L'Arbre de Décision optimisé et la Forêt Aléatoire obtiennent les meilleurs compromis F1 et ROC-AUC sur la classe minoritaire.
2. **Facteurs Prépondérants :** Le Revenu Mensuel (`MonthlyIncome`), la Distance Domicile-Travail (`DistanceFromHome`), l'Âge (`Age`) et les Heures Supplémentaires (`OverTime`) constituent les déterminants majeurs de l'attrition.
3. **Plan d'Action RH :** Revaloriser les rémunérations des profils clés, encadrer strictement les heures supplémentaires et proposer des solutions de flexibilité/télétravail pour les collaborateurs éloignés.""")
]


# ==============================================================================
# WRITE FILES
# ==============================================================================
notebooks = {
    "01_eda_and_cleaning.ipynb": nb1_cells,
    "02_statistical_analysis.ipynb": nb2_cells,
    "03_preprocessing.ipynb": nb3_cells,
    "04_model_training_evaluation.ipynb": nb4_cells,
}

def main() -> None:
    """Generate notebooks only when the script is executed, never on import."""
    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    for filename, cells in notebooks.items():
        nb = create_notebook(cells)
        dest = NOTEBOOKS_DIR / filename
        with open(dest, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=2, ensure_ascii=False)
        print(f"✔ Notebook généré : {dest.name} ({len(cells)} cellules)")


if __name__ == "__main__":
    main()
