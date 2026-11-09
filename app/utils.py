"""Shared utilities for the Streamlit application."""
import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from typing import Optional

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA = DATA_DIR / "raw" / "Employee_Attrition_DataSet.csv"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# ── Feature definitions ────────────────────────────────────────────────
NUMERICAL_FEATURES = [
    "Age", "JobLevel", "YearsAtCompany", "MonthlyIncome",
    "JobSatisfaction", "WorkLifeBalance", "DistanceFromHome",
    "PerformanceRating", "TrainingHoursLastYear"
]
CATEGORICAL_FEATURES = ["Department", "OverTime", "PromotionLast5Years"]
TARGET = "Attrition"
ID_COL = "EmployeeID"
ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

MODEL_NAMES = {
    "ZeroR": "Zero-R (Baseline)",
    "KNN": "K-NN",
    "DecisionTree": "Arbre de Décision",
    "RandomForest": "Forêt Aléatoire",
    "NaiveBayes": "Naive Bayes",
    "SVM": "SVM",
}

DEPARTMENT_LIST = ["Sales", "Engineering", "Operations", "Marketing", "Finance", "IT", "HR"]

# ── Custom CSS ─────────────────────────────────────────────────────────
def inject_custom_css():
    """Inject custom CSS styles into the Streamlit app."""
    st.markdown("""
    <style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 0.75rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-card h3 { font-size: 0.9rem; margin: 0; opacity: 0.9; }
    .metric-card h1 { font-size: 2rem; margin: 0.3rem 0 0 0; }
    .metric-card-green { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }
    .metric-card-blue { background: linear-gradient(135deg, #2193b0 0%, #6dd5ed 100%); }
    .metric-card-orange { background: linear-gradient(135deg, #f2994a 0%, #f2c94c 100%); }
    .metric-card-red { background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%); }
    .metric-card-purple { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    .risk-badge {
        display: inline-block;
        padding: 0.4rem 1.2rem;
        border-radius: 2rem;
        font-weight: bold;
        font-size: 1.1rem;
    }
    .risk-low { background-color: #d4edda; color: #155724; }
    .risk-medium { background-color: #fff3cd; color: #856404; }
    .risk-high { background-color: #f8d7da; color: #721c24; }
    div[data-testid="stSidebar"] > div:first-child {
        padding-top: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)


def metric_card(title: str, value, css_class: str = ""):
    """Render a styled metric card."""
    cls = f"metric-card {css_class}" if css_class else "metric-card"
    st.markdown(
        f'<div class="{cls}"><h3>{title}</h3><h1>{value}</h1></div>',
        unsafe_allow_html=True,
    )


# ── Data loading ───────────────────────────────────────────────────────
@st.cache_data
def load_data() -> Optional[pd.DataFrame]:
    """Load the raw dataset. Returns None if file not found."""
    if not RAW_DATA.exists():
        return None
    return pd.read_csv(RAW_DATA)


@st.cache_resource
def load_model(name: str = "best_model"):
    """Load a saved model pipeline. Returns None on failure."""
    path = MODELS_DIR / f"{name}.joblib"
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


@st.cache_data
def load_json(filename: str) -> Optional[dict]:
    """Load a JSON results file. Returns None on failure."""
    path = RESULTS_DIR / filename
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# ── Risk level ─────────────────────────────────────────────────────────
def get_risk_level(proba: float) -> tuple:
    """Return (label, css_class, emoji) based on attrition probability."""
    if proba < 0.3:
        return "Faible", "risk-low", "🟢"
    elif proba < 0.6:
        return "Modéré", "risk-medium", "🟡"
    else:
        return "Élevé", "risk-high", "🔴"


def get_risk_color(proba: float) -> str:
    """Return a hex color based on attrition probability."""
    if proba < 0.3:
        return "#28a745"
    elif proba < 0.6:
        return "#ffc107"
    else:
        return "#dc3545"


# ── Recommendations ───────────────────────────────────────────────────
def generate_recommendations(row: dict, proba: float) -> list:
    """Generate rule-based retention recommendations."""
    recs = []
    if row.get("OverTime") == "Yes":
        recs.append("⏰ Réduire les heures supplémentaires — l'overtime est un facteur majeur de départ.")
    if row.get("JobSatisfaction", 5) <= 2:
        recs.append("😊 Améliorer la satisfaction au travail — envisager un entretien individuel.")
    if row.get("WorkLifeBalance", 5) <= 2:
        recs.append("⚖️ Améliorer l'équilibre vie pro/perso — proposer du télétravail ou des horaires flexibles.")
    if row.get("YearsAtCompany", 0) <= 2:
        recs.append("🎯 Renforcer l'intégration — les employés récents sont plus susceptibles de partir.")
    if row.get("MonthlyIncome", 99999) < 5000:
        recs.append("💰 Réviser la rémunération — le salaire est en dessous de la moyenne.")
    if row.get("DistanceFromHome", 0) > 25:
        recs.append("🏠 Proposer du télétravail — la distance domicile-travail est élevée.")
    if row.get("PromotionLast5Years") == "No" and row.get("YearsAtCompany", 0) >= 5:
        recs.append("📈 Envisager une promotion — aucune promotion depuis 5+ ans.")
    if row.get("TrainingHoursLastYear", 99) < 10:
        recs.append("📚 Augmenter les heures de formation — le développement professionnel fidélise.")
    if not recs:
        recs.append("✅ Profil stable — continuer les bonnes pratiques de rétention.")
    return recs


# ── Sidebar ────────────────────────────────────────────────────────────
def render_sidebar_footer():
    """Render the sidebar footer with disclaimer."""
    st.sidebar.markdown("---")
    st.sidebar.caption(
        "📊 **Projet Data Mining**\n\n"
        "Prédiction d'attrition des employés\n\n"
        "⚠️ Données synthétiques — Usage académique uniquement"
    )


# ── Missing artifact warning ──────────────────────────────────────────
def show_missing_artifact_warning(artifact_name: str, fix_command: str = ""):
    """Show a warning when an artifact is missing."""
    msg = f"⚠️ **{artifact_name}** non trouvé."
    if fix_command:
        msg += f"\n\nPour générer cet artefact, exécutez :\n```\n{fix_command}\n```"
    st.warning(msg)
