"""Home page — Landing page for the Employee Attrition prediction app."""
from pathlib import Path
import sys

import streamlit as st

# Ensure project root is on sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

st.set_page_config(
    page_title="Attrition RH — Accueil",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

from app.utils import (
    load_css,
    load_data,
    kpi_card,
    section_header,
    render_sidebar,
    show_missing_artifact_warning,
    PRIMARY_COLOR,
    SECONDARY_COLOR,
    COLOR_NO,
    COLOR_YES,
    COLOR_WARN,
)

# Inject custom stylesheet
load_css()

# Render shared sidebar
render_sidebar(df=None, show_filters=False)

# ── Hero Banner ────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div style="background: linear-gradient(135deg, rgba(79, 70, 229, 0.08) 0%, rgba(6, 182, 212, 0.08) 100%);
                border: 1px solid rgba(79, 70, 229, 0.2); border-radius: 1rem; padding: 1.8rem 2rem; margin-bottom: 1.5rem;">
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 1rem;">
            <div>
                <span style="display: inline-block; padding: 0.25rem 0.75rem; border-radius: 9999px; font-size: 0.75rem;
                             font-weight: 600; background: #EEF2FF; color: {PRIMARY_COLOR}; margin-bottom: 0.5rem;">
                    Data Mining • Track A : Classification Supervisée
                </span>
                <h1 style="font-size: 2.1rem; font-weight: 800; margin: 0; color: #111827; letter-spacing: -0.02em;">
                    💼 Prédiction de l'Attrition des Employés
                </h1>
                <p style="font-size: 1.05rem; color: #4B5563; margin-top: 0.5rem; max-width: 820px; line-height: 1.5;">
                    Plateforme analytique et prédictive conçue pour anticiper les départs volontaires,
                    identifier les facteurs clés d'engagement RH et recommander des plans d'action ciblés.
                </p>
            </div>
            <div style="background: white; border: 1px solid #E5E7EB; border-radius: 0.75rem; padding: 0.9rem 1.4rem; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                <span style="font-size: 0.75rem; color: #6B7280; text-transform: uppercase; font-weight: 600; letter-spacing: 0.05em;">Base Étudiée</span>
                <div style="font-size: 1.6rem; font-weight: 700; color: {PRIMARY_COLOR};">10 000 RH</div>
                <span style="font-size: 0.75rem; color: #10B981; font-weight: 500;">Jeu de données vérifié</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── KPI Cards ──────────────────────────────────────────────────────────
df = load_data()

if df is not None:
    total_employees = len(df)
    attrition_count = (df["Attrition"] == "Yes").sum()
    attrition_rate = (attrition_count / total_employees) * 100
    avg_salary = df["MonthlyIncome"].mean()
    avg_tenure = df["YearsAtCompany"].mean()
    avg_satisfaction = df["JobSatisfaction"].mean()

    section_header(
        "Indicateurs Clés Globaux",
        "Aperçu statistique de la population des employés avant modélisation",
        icon="📌",
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Effectif Total", f"{total_employees:,}", border_color=PRIMARY_COLOR, help_text="Total des collaborateurs analysés")
    with c2:
        kpi_card("Taux d'Attrition", f"{attrition_rate:.1f}%", delta=f"{attrition_count} départs", delta_type="negative", border_color=COLOR_YES, help_text="Pourcentage d'employés ayant quitté l'entreprise")
    with c3:
        kpi_card("Salaire Moyen", f"{avg_salary:,.0f} MAD", delta="Mensuel brut", border_color=COLOR_NO, help_text="Rémunération mensuelle moyenne")
    with c4:
        kpi_card("Ancienneté Moyenne", f"{avg_tenure:.1f} ans", delta=f"Satisfaction {avg_satisfaction:.1f}/5", border_color=COLOR_WARN, help_text="Nombre moyen d'années au sein de la structure")
else:
    show_missing_artifact_warning(
        "Jeu de données (Employee_Attrition_DataSet.csv)",
        "Placez le fichier dans data/raw/",
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── Navigation Cards ───────────────────────────────────────────────────
section_header(
    "Modules de l'Application",
    "Explorez les données, évaluez les modèles et effectuez des prédictions en direct",
    icon="🗂️",
)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <div class="nav-card">
            <div>
                <div class="nav-card-icon">📊</div>
                <div class="nav-card-title">1. Dashboard EDA</div>
                <div class="nav-card-desc">
                    Analyse exploratoire interactive : distributions, corrélations, tris croisés et filtres dynamiques multicritères.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.page_link("pages/1_Dashboard.py", label="Explorer le Dashboard", icon="📊", use_container_width=True)

with col2:
    st.markdown(
        """
        <div class="nav-card">
            <div>
                <div class="nav-card-icon">🔮</div>
                <div class="nav-card-title">2. Prédiction Individuelle</div>
                <div class="nav-card-desc">
                    Formulaire interactif avec jauge de risque, recommandations d'action RH et simulation « What-if » en temps réel.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.page_link("pages/2_Prediction.py", label="Tester une Prédiction", icon="🔮", use_container_width=True)

with col3:
    st.markdown(
        """
        <div class="nav-card">
            <div>
                <div class="nav-card-icon">🏆</div>
                <div class="nav-card-title">3. Performance Modèles</div>
                <div class="nav-card-desc">
                    Benchmark rigoureux de 6 algorithmes (Zero-R, KNN, Decision Tree, Random Forest, Naive Bayes, SVM).
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.page_link("pages/3_Performance_Modeles.py", label="Comparer les Modèles", icon="🏆", use_container_width=True)

st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

col4, col5, col6 = st.columns(3)

with col4:
    st.markdown(
        """
        <div class="nav-card">
            <div>
                <div class="nav-card-icon">🔍</div>
                <div class="nav-card-title">4. Facteurs de Démission</div>
                <div class="nav-card-desc">
                    Explicabilité des algorithmes : importance des variables (Feature Importance) et analyse des contributions SHAP.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.page_link("pages/4_Facteurs_Demission.py", label="Analyser les Facteurs", icon="🔍", use_container_width=True)

with col5:
    st.markdown(
        """
        <div class="nav-card">
            <div>
                <div class="nav-card-icon">📦</div>
                <div class="nav-card-title">5. Prédiction en Lot</div>
                <div class="nav-card-desc">
                    Import de fichiers CSV pour prédictions massives avec tableau interactif, jauge colorée et export Excel/CSV.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.page_link("pages/5_Prediction_en_lot.py", label="Lancer un Lot", icon="📦", use_container_width=True)

with col6:
    st.markdown(
        """
        <div class="nav-card">
            <div>
                <div class="nav-card-icon">📖</div>
                <div class="nav-card-title">6. À Propos du Projet</div>
                <div class="nav-card-desc">
                    Détails du pipeline Data Mining, méthodologie scientifique, dictionnaire des données et mentions éthiques.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.page_link("pages/6_A_Propos.py", label="Consulter la Documentation", icon="📖", use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Pipeline Workflow ──────────────────────────────────────────────────
section_header(
    "Pipeline Scientifique de Data Mining",
    "Méthodologie standard CRISP-DM appliquée au jeu de données RH",
    icon="⚙️",
)

st.markdown(
    """
    <div style="background: white; border: 1px solid #E5E7EB; border-radius: 0.75rem; padding: 1.25rem; overflow-x: auto;">
        <table style="width: 100%; border-collapse: collapse; font-size: 0.9rem;">
            <thead>
                <tr style="border-bottom: 2px solid #E5E7EB; text-align: left; color: #4B5563;">
                    <th style="padding: 0.75rem 1rem;">Phase</th>
                    <th style="padding: 0.75rem 1rem;">Opérations Réalisées</th>
                    <th style="padding: 0.75rem 1rem;">Outils & Garanties</th>
                </tr>
            </thead>
            <tbody>
                <tr style="border-bottom: 1px solid #F3F4F6;">
                    <td style="padding: 0.75rem 1rem; font-weight: 600; color: #4F46E5;">1. Audit & Exploration</td>
                    <td style="padding: 0.75rem 1rem;">Contrôle qualité (10 000 observations), distribution cible, détection des anomalies</td>
                    <td style="padding: 0.75rem 1rem; color: #6B7280;">Pandas, Plotly, Seaborn</td>
                </tr>
                <tr style="border-bottom: 1px solid #F3F4F6;">
                    <td style="padding: 0.75rem 1rem; font-weight: 600; color: #4F46E5;">2. Tests Statistiques</td>
                    <td style="padding: 0.75rem 1rem;">Tests d'hypothèse (Chi², Mann-Whitney U, Spearman) avec correction Benjamini-Hochberg</td>
                    <td style="padding: 0.75rem 1rem; color: #6B7280;">SciPy, Statsmodels</td>
                </tr>
                <tr style="border-bottom: 1px solid #F3F4F6;">
                    <td style="padding: 0.75rem 1rem; font-weight: 600; color: #4F46E5;">3. Pré-traitement Rigoureux</td>
                    <td style="padding: 0.75rem 1rem;">StandardScaler (numérique) + OneHotEncoder (catégoriel) sans fuite de données</td>
                    <td style="padding: 0.75rem 1rem; color: #6B7280;">Scikit-Learn ColumnTransformer Pipeline</td>
                </tr>
                <tr style="border-bottom: 1px solid #F3F4F6;">
                    <td style="padding: 0.75rem 1rem; font-weight: 600; color: #4F46E5;">4. Modélisation & Tuning</td>
                    <td style="padding: 0.75rem 1rem;">Entraînement de 6 algorithmes avec validation croisée stratifiée 5-Fold et recherche d'hyperparamètres</td>
                    <td style="padding: 0.75rem 1rem; color: #6B7280;">GridSearchCV, RandomizedSearchCV</td>
                </tr>
                <tr>
                    <td style="padding: 0.75rem 1rem; font-weight: 600; color: #4F46E5;">5. Déploiement & IHM</td>
                    <td style="padding: 0.75rem 1rem;">Interface Web Streamlit dynamique, simulation What-if, rapports et explications SHAP</td>
                    <td style="padding: 0.75rem 1rem; color: #6B7280;">Streamlit, Plotly, SHAP</td>
                </tr>
            </tbody>
        </table>
    </div>
    """,
    unsafe_allow_html=True,
)
