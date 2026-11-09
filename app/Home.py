"""Home page — Landing page for the Employee Attrition prediction app."""
import streamlit as st
import sys
from pathlib import Path

# Ensure project root is on sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

st.set_page_config(
    page_title="Attrition RH — Accueil",
    page_icon="🏠",
    layout="wide",
)

from app.utils import (
    inject_custom_css,
    render_sidebar_footer,
    load_data,
    metric_card,
    show_missing_artifact_warning,
)

inject_custom_css()

# ── Sidebar ────────────────────────────────────────────────────────────
st.sidebar.title("🏠 Navigation")
st.sidebar.info("Utilisez le menu latéral pour naviguer entre les pages.")
render_sidebar_footer()

# ── Header ─────────────────────────────────────────────────────────────
st.title("🏠 Prédiction de l'Attrition des Employés")
st.markdown(
    """
    > **Projet de Data Mining — Classification supervisée (Track A)**
    >
    > Cette application analyse les facteurs d'attrition des employés et propose des prédictions
    > basées sur des modèles de machine learning entraînés sur un jeu de données de 10 000 employés.
    """
)

st.markdown("---")

# ── KPI cards ──────────────────────────────────────────────────────────
df = load_data()

if df is not None:
    total_employees = len(df)
    attrition_count = (df["Attrition"] == "Yes").sum()
    attrition_rate = attrition_count / total_employees * 100
    avg_salary = df["MonthlyIncome"].mean()
    avg_tenure = df["YearsAtCompany"].mean()

    st.subheader("📌 Indicateurs Clés")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        metric_card("Total Employés", f"{total_employees:,}", "metric-card-blue")
    with col2:
        metric_card("Taux d'Attrition", f"{attrition_rate:.1f}%", "metric-card-red")
    with col3:
        metric_card("Salaire Moyen", f"{avg_salary:,.0f} MAD", "metric-card-green")
    with col4:
        metric_card("Ancienneté Moyenne", f"{avg_tenure:.1f} ans", "metric-card-orange")

    st.markdown("<br>", unsafe_allow_html=True)
else:
    show_missing_artifact_warning(
        "Jeu de données",
        "Placez le fichier Employee_Attrition_DataSet.csv dans data/raw/",
    )

# ── Pipeline overview ─────────────────────────────────────────────────
st.subheader("📋 Vue d'ensemble du Pipeline")
st.markdown(
    """
    | Étape | Description |
    |-------|-------------|
    | 1. **Chargement des données** | Import du fichier CSV (10 000 lignes × 14 colonnes) |
    | 2. **Exploration (EDA)** | Analyse statistique, visualisations, détection de déséquilibre |
    | 3. **Pré-traitement** | Encodage des variables catégorielles, normalisation, SMOTE |
    | 4. **Modélisation** | Entraînement de 6 modèles (Zero-R, KNN, Arbre de Décision, Forêt Aléatoire, Naive Bayes, SVM) |
    | 5. **Évaluation** | Accuracy, Precision, Recall, F1-Score, AUC-ROC, Validation croisée |
    | 6. **Déploiement** | Application Streamlit interactive avec prédiction en temps réel |
    """
)

st.markdown("---")

# ── Page navigation cards ─────────────────────────────────────────────
st.subheader("🗂️ Pages de l'Application")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(
        """
        #### 📊 Dashboard EDA
        Explorez les données avec des graphiques interactifs,
        des filtres et des statistiques descriptives.
        """
    )
    st.page_link("pages/1_Dashboard.py", label="Ouvrir le Dashboard", icon="📊")

with col2:
    st.markdown(
        """
        #### 🔮 Prédiction Individuelle
        Saisissez le profil d'un employé et obtenez une prédiction
        de risque d'attrition en temps réel.
        """
    )
    st.page_link("pages/2_Prediction.py", label="Faire une Prédiction", icon="🔮")

with col3:
    st.markdown(
        """
        #### 🏆 Performance des Modèles
        Comparez les performances des 6 modèles entraînés
        avec des métriques détaillées et courbes ROC.
        """
    )
    st.page_link("pages/3_Performance_Modeles.py", label="Voir les Performances", icon="🏆")

with col4:
    st.markdown(
        """
        #### 🔍 Facteurs de Démission
        Identifiez les variables les plus influentes sur
        l'attrition grâce à l'importance des features et SHAP.
        """
    )
    st.page_link("pages/4_Facteurs_Demission.py", label="Analyser les Facteurs", icon="🔍")

with col5:
    st.markdown(
        """
        #### 📦 Prédiction en Lot
        Chargez un fichier CSV de plusieurs employés et obtenez
        des prédictions groupées avec export des résultats.
        """
    )
    st.page_link("pages/5_Prediction_en_lot.py", label="Prédiction en Lot", icon="📦")

st.markdown("---")

# ── Project architecture ──────────────────────────────────────────────
st.subheader("🔧 Architecture du Projet")
st.code(
    """
data-mining project/
├── .streamlit/
│   └── config.toml              # Configuration Streamlit
├── app/
│   ├── Home.py                  # Page d'accueil
│   ├── utils.py                 # Utilitaires partagés
│   └── pages/
│       ├── 1_Dashboard.py       # Dashboard EDA
│       ├── 2_Prediction.py      # Prédiction individuelle
│       ├── 3_Performance_Modeles.py  # Comparaison des modèles
│       ├── 4_Facteurs_Demission.py   # Importance des variables
│       └── 5_Prediction_en_lot.py    # Prédiction en lot
├── data/
│   └── raw/
│       └── Employee_Attrition_DataSet.csv
├── models/
│   ├── best_model.joblib        # Meilleur modèle
│   └── *_pipeline.joblib        # Pipelines entraînés
├── results/
│   ├── metrics.json             # Métriques de performance
│   ├── roc_data.json            # Données courbes ROC
│   ├── confusion_matrices.json  # Matrices de confusion
│   ├── feature_importances.json # Importance des variables
│   ├── best_hyperparameters.json# Hyperparamètres optimaux
│   └── cv_results_summary.json  # Résultats validation croisée
├── src/
│   ├── app.py                   # Point d'entrée
│   └── preprocessing.py         # Pré-traitement
├── notebooks/                   # Notebooks Jupyter
└── tests/
    └── test_app_smoke.py        # Tests de fumée
    """,
    language=None,
)

# ── Disclaimer ─────────────────────────────────────────────────────────
st.markdown("---")
st.info(
    "⚠️ **Avertissement** : Ce projet utilise des données synthétiques générées à des fins "
    "pédagogiques. Les prédictions ne doivent pas être utilisées pour des décisions réelles "
    "en matière de ressources humaines. Usage académique uniquement."
)
