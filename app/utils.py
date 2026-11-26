"""Shared utilities, UI components and theming for the Streamlit application."""
import json
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import load_bundle as _load_bundle  # noqa: E402
from src.preprocessing import (  # noqa: E402
    CATEGORICAL_FEATURES as _CATEGORICAL_FEATURES,
    FEATURE_COLS,
    ID_COL,
    NUMERICAL_FEATURES as _NUMERICAL_FEATURES,
    TARGET,
)

# ── Paths ──────────────────────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA = DATA_DIR / "raw" / "Employee_Attrition_DataSet.csv"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
CSS_PATH = Path(__file__).resolve().parent / "style.css"

# ── Design Tokens & Theming ────────────────────────────────────────────
PRIMARY_COLOR = "#5B67F1"        # Royal periwinkle
SECONDARY_COLOR = "#34BFA3"      # Sea green
COLOR_NO = "#20B486"             # Emerald (Stayed / Low risk)
COLOR_YES = "#E85D75"            # Coral rose (Left / High risk)
COLOR_WARN = "#F2A93B"           # Amber (Moderate risk)
COLOR_MAP = {"No": COLOR_NO, "Yes": COLOR_YES}

PLOTLY_TEMPLATE = "plotly_white"
PLOTLY_COLORWAY = [
    "#5B67F1",  # Periwinkle
    "#34BFA3",  # Sea green
    "#E85D75",  # Coral
    "#F2A93B",  # Amber
    "#7C5CE7",  # Violet
    "#4BA3C7",  # Sky
    "#97A4C2",  # Mist blue
    "#64748B",  # Slate
]

# ── Feature definitions (single source of truth: src.preprocessing) ────
NUMERICAL_FEATURES = list(_NUMERICAL_FEATURES)
CATEGORICAL_FEATURES = list(_CATEGORICAL_FEATURES)
ALL_FEATURES = list(FEATURE_COLS)  # alias conservé pour les pages existantes

MODEL_NAMES = {
    "ZeroR": "Zero-R (Baseline)",
    "KNN": "K-NN",
    "DecisionTree": "Arbre de Décision",
    "RandomForest": "Forêt Aléatoire",
    "NaiveBayes": "Naive Bayes",
    "SVM": "SVM",
}

DEPARTMENT_LIST = ["Sales", "Engineering", "Operations", "Marketing", "Finance", "IT", "HR"]


def get_plotly_template() -> str:
    """Return a Plotly template that matches the current app theme."""
    return "plotly_dark" if st.session_state.get("dark_mode", False) else PLOTLY_TEMPLATE


# ── CSS Loading ────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _read_app_css() -> str:
    """Read the stylesheet once per process instead of every Streamlit rerun."""
    if CSS_PATH.exists():
        return CSS_PATH.read_text(encoding="utf-8")
    return ""


def load_css() -> None:
    """Load custom CSS from app/style.css and inject into Streamlit."""
    css = _read_app_css()
    if css:
        if st.session_state.get("dark_mode", False):
            css += """
            :root {
                color-scheme: dark;
                --primary-light: #312E81;
                --bg-card: #1E293B;
                --border-color: #334155;
                --text-main: #F8FAFC;
                --text-muted: #CBD5E1;
                --risk-low-bg: rgba(6, 78, 59, 0.4);
                --risk-low-text: #6EE7B7;
                --risk-low-border: #059669;
                --risk-med-bg: rgba(120, 53, 15, 0.4);
                --risk-med-text: #FCD34D;
                --risk-med-border: #D97706;
                --risk-high-bg: rgba(127, 29, 29, 0.4);
                --risk-high-text: #FCA5A5;
                --risk-high-border: #DC2626;
                --background-color: #0E1117;
                --secondary-background-color: #1E293B;
                --text-color: #F8FAFC;
            }
            html, body, .stApp, [data-testid="stAppViewContainer"],
            [data-testid="stHeader"], [data-testid="stSidebar"] {
                background-color: #0E1117 !important;
                color: #F8FAFC;
            }
            [data-testid="stSidebar"] {
                background-color: #111827 !important;
            }
            [data-testid="stMarkdownContainer"] p,
            [data-testid="stMarkdownContainer"] li,
            [data-testid="stWidgetLabel"] p,
            [data-testid="stCaptionContainer"] {
                color: #E2E8F0;
            }
            [data-testid="stMarkdownContainer"] h1,
            [data-testid="stMarkdownContainer"] h2,
            [data-testid="stMarkdownContainer"] h3,
            [data-testid="stMarkdownContainer"] h4 {
                color: #F8FAFC !important;
            }
            div[style*="background: white"],
            div[style*="background:white"],
            div[style*="background-color: white"] {
                background: #1E293B !important;
                color: #F8FAFC !important;
                border-color: #334155 !important;
            }
            input, textarea, [data-baseweb="select"] > div,
            [data-baseweb="popover"], [data-baseweb="menu"] {
                background-color: #1E293B !important;
                color: #F8FAFC !important;
                border-color: #475569 !important;
            }
            [data-testid="stDataFrame"], [data-testid="stTable"] {
                background-color: #1E293B;
            }
            """
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        .kpi-card { background:#f9fafb; border-left:4px solid #4F46E5; padding:1rem; border-radius:0.75rem; }
        .risk-badge { padding:0.3rem 0.8rem; border-radius:1rem; font-weight:600; }
        .risk-low { background:#d1fae5; color:#065f46; }
        .risk-medium { background:#fef3c7; color:#92400e; }
        .risk-high { background:#fee2e2; color:#991b1b; }
        </style>
        """, unsafe_allow_html=True)


def inject_custom_css() -> None:
    """Backwards compatibility alias for load_css()."""
    load_css()


# ── Reusable UI Components ─────────────────────────────────────────────
def kpi_card(
    titre: str,
    valeur: Any,
    delta: Optional[str] = None,
    delta_type: str = "neutral",
    border_color: str = PRIMARY_COLOR,
    help_text: Optional[str] = None,
) -> None:
    """Render a modern KPI card with a colored left border and subtle shadow."""
    delta_html = ""
    if delta:
        delta_class = "neutral"
        if delta_type == "positive" or (isinstance(delta, str) and delta.startswith("+")):
            delta_class = "positive"
        elif delta_type == "negative" or (isinstance(delta, str) and delta.startswith("-")):
            delta_class = "negative"
        delta_html = f'<div class="kpi-delta {delta_class}">{delta}</div>'

    title_attr = f'title="{help_text}"' if help_text else ""
    st.markdown(
        f"""
        <div class="kpi-card" style="border-left-color: {border_color} !important;" {title_attr}>
            <div class="kpi-title">{titre}</div>
            <div class="kpi-value">{valeur}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(title: str, value: Any, css_class: str = "") -> None:
    """Backwards compatibility metric card using the new theme style."""
    border_color = PRIMARY_COLOR
    if "green" in css_class:
        border_color = COLOR_NO
    elif "red" in css_class:
        border_color = COLOR_YES
    elif "orange" in css_class or "amber" in css_class:
        border_color = COLOR_WARN
    elif "blue" in css_class or "cyan" in css_class:
        border_color = SECONDARY_COLOR

    kpi_card(title, value, border_color=border_color)


def risk_badge(probabilite: float, large: bool = False) -> str:
    """Return an HTML badge string with appropriate color and label based on probability."""
    p_pct = probabilite * 100 if probabilite <= 1.0 else probabilite
    size_cls = "large" if large else ""

    if p_pct < 30.0:
        return f'<span class="risk-badge risk-low {size_cls}">🟢 Risque Faible ({p_pct:.1f}%)</span>'
    elif p_pct < 60.0:
        return f'<span class="risk-badge risk-medium {size_cls}">🟡 Risque Modéré ({p_pct:.1f}%)</span>'
    else:
        return f'<span class="risk-badge risk-high {size_cls}">🔴 Risque Élevé ({p_pct:.1f}%)</span>'


def section_header(titre: str, sous_titre: Optional[str] = None, icon: Optional[str] = None) -> None:
    """Render a styled section header with title and subtitle."""
    icon_html = f"<span>{icon}</span> " if icon else ""
    sub_html = f'<p class="section-header-subtitle">{sous_titre}</p>' if sous_titre else ""
    st.markdown(
        f"""
        <div class="section-header-container">
            <h3 class="section-header-title">{icon_html}{titre}</h3>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def gauge_chart(probabilite: float, title: str = "Risque d'Attrition") -> go.Figure:
    """Generate an interactive Plotly gauge chart for attrition risk."""
    p_pct = probabilite * 100 if probabilite <= 1.0 else probabilite

    if p_pct < 30.0:
        bar_color = COLOR_NO
    elif p_pct < 60.0:
        bar_color = COLOR_WARN
    else:
        bar_color = COLOR_YES

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=p_pct,
            number={"suffix": "%", "font": {"size": 38, "color": "#111827", "family": "sans-serif"}},
            title={"text": f"<b>{title}</b>", "font": {"size": 16, "color": "#4B5563"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#9CA3AF"},
                "bar": {"color": bar_color, "thickness": 0.28},
                "bgcolor": "white",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 30], "color": "#D1FAE5"},
                    {"range": [30, 60], "color": "#FEF3C7"},
                    {"range": [60, 100], "color": "#FEE2E2"},
                ],
                "threshold": {
                    "line": {"color": bar_color, "width": 4},
                    "thickness": 0.8,
                    "value": p_pct,
                },
            },
        )
    )

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=260,
        margin=dict(t=40, b=10, l=30, r=30),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "sans-serif"},
    )
    return fig


# ── Data Loading & Model Loading ───────────────────────────────────────
@st.cache_data
def load_data() -> Optional[pd.DataFrame]:
    """Load the raw dataset. Returns None if file not found."""
    if not RAW_DATA.exists():
        return None
    return pd.read_csv(RAW_DATA)


@st.cache_resource
def load_bundle(name: str = "best_model"):
    """Load the trained Pipeline bundle, including the optimized threshold."""
    path = MODELS_DIR / f"{name}.joblib"
    if not path.exists():
        return None
    try:
        return _load_bundle(path)
    except Exception:
        return None


@st.cache_resource
def load_model(name: str = "best_model"):
    """Compatibility helper: return the Pipeline held by the bundle."""
    bundle = load_bundle(name)
    return None if bundle is None else bundle["model"]


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


# ── Risk Helpers ───────────────────────────────────────────────────────
def get_risk_level(proba: float) -> Tuple[str, str, str]:
    """Return (label, css_class, emoji) based on attrition probability."""
    p_pct = proba * 100 if proba <= 1.0 else proba
    if p_pct < 30.0:
        return "Faible", "risk-low", "🟢"
    elif p_pct < 60.0:
        return "Modéré", "risk-medium", "🟡"
    else:
        return "Élevé", "risk-high", "🔴"


def get_risk_color(proba: float) -> str:
    """Return a hex color based on attrition probability."""
    p_pct = proba * 100 if proba <= 1.0 else proba
    if p_pct < 30.0:
        return COLOR_NO
    elif p_pct < 60.0:
        return COLOR_WARN
    else:
        return COLOR_YES


# ── Recommendations ───────────────────────────────────────────────────
def generate_recommendations(row: dict, proba: float) -> List[str]:
    """Generate rule-based retention recommendations."""
    recs = []
    if row.get("OverTime") == "Yes":
        recs.append("⏰ **Réduire les heures supplémentaires** : l'overtime est un facteur majeur de départ.")
    if row.get("JobSatisfaction", 5) <= 2:
        recs.append("😊 **Améliorer la satisfaction au travail** : organiser un entretien individuel d'écoute.")
    if row.get("WorkLifeBalance", 5) <= 2:
        recs.append("⚖️ **Équilibre vie pro/perso** : proposer du télétravail ou des horaires aménagés.")
    if row.get("YearsAtCompany", 0) <= 2:
        recs.append("🎯 **Renforcer le parcours d'intégration** : les nouvelles recrues sont les plus vulnérables.")
    if row.get("MonthlyIncome", 99999) < 5500:
        recs.append("💰 **Réviser la grille salariale** : le salaire se situe sous la moyenne observée.")
    if row.get("DistanceFromHome", 0) > 25:
        recs.append("🏠 **Aménager le transport / distanciel** : le temps de trajet domicile-travail est élevé.")
    if row.get("PromotionLast5Years") == "No" and row.get("YearsAtCompany", 0) >= 5:
        recs.append("📈 **Planifier une évolution de carrière** : aucune promotion enregistrée depuis 5 ans.")
    if row.get("TrainingHoursLastYear", 99) < 10:
        recs.append("📚 **Proposer des formations** : le développement des compétences fidélise les talents.")
    if not recs:
        recs.append("✅ **Profil stable** : maintenir les actions régulières de reconnaissance et d'écoute.")
    return recs


# ── Global Sidebar & Filter Session State ──────────────────────────────
def init_session_state(df: Optional[pd.DataFrame] = None) -> None:
    """Initialize session state filters if not present."""
    if "global_departments" not in st.session_state:
        if df is not None and "Department" in df.columns:
            st.session_state.global_departments = sorted(df["Department"].unique().tolist())
        else:
            st.session_state.global_departments = DEPARTMENT_LIST

    if "global_overtime" not in st.session_state:
        st.session_state.global_overtime = "Tous"

    if "global_promotion" not in st.session_state:
        st.session_state.global_promotion = "Tous"

    if "global_age_range" not in st.session_state:
        if df is not None and "Age" in df.columns:
            st.session_state.global_age_range = (int(df["Age"].min()), int(df["Age"].max()))
        else:
            st.session_state.global_age_range = (20, 65)

    if "global_income_range" not in st.session_state:
        if df is not None and "MonthlyIncome" in df.columns:
            st.session_state.global_income_range = (int(df["MonthlyIncome"].min()), int(df["MonthlyIncome"].max()))
        else:
            st.session_state.global_income_range = (2000, 20000)


@st.cache_data(show_spinner=False)
def filter_employee_data(
    df: pd.DataFrame,
    departments: Tuple[str, ...],
    overtime: str,
    age_range: Tuple[int, int],
    income_range: Tuple[int, int],
) -> pd.DataFrame:
    """Apply sidebar filters once and reuse the cached result across reruns."""
    filtered_df = df
    if departments:
        filtered_df = filtered_df[filtered_df["Department"].isin(departments)]
    if overtime == "Oui":
        filtered_df = filtered_df[filtered_df["OverTime"] == "Yes"]
    elif overtime == "Non":
        filtered_df = filtered_df[filtered_df["OverTime"] == "No"]
    return filtered_df[
        (filtered_df["Age"] >= age_range[0])
        & (filtered_df["Age"] <= age_range[1])
        & (filtered_df["MonthlyIncome"] >= income_range[0])
        & (filtered_df["MonthlyIncome"] <= income_range[1])
    ].copy()


def render_sidebar(df: Optional[pd.DataFrame] = None, show_filters: bool = True) -> Optional[pd.DataFrame]:
    """Render consistent sidebar branding, optional global filters, and footer."""
    init_session_state(df)

    st.sidebar.markdown(
        """
        <div class="sidebar-brand">
            <span style="font-size: 1.6rem;">💼</span>
            <div>
                <h3 class="sidebar-brand-title">Attrition RH</h3>
                <p class="sidebar-brand-sub">Data Mining & Classification</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.checkbox(
        "Mode sombre",
        key="dark_mode",
        help="Activer ou désactiver le thème sombre de l'application",
    )

    filtered_df = df.copy() if df is not None else None

    if show_filters and df is not None:
        st.sidebar.markdown("##### 🔎 Filtres Globaux")

        with st.sidebar.expander("⚙️ Paramètres des filtres", expanded=True):
            # Department filter
            all_depts = sorted(df["Department"].unique().tolist())
            st.session_state.global_departments = st.multiselect(
                "Départements",
                options=all_depts,
                default=st.session_state.global_departments,
                help="Filtrer par un ou plusieurs départements",
            )

            # Overtime filter
            ot_options = ["Tous", "Oui", "Non"]
            ot_idx = ot_options.index(st.session_state.global_overtime) if st.session_state.global_overtime in ot_options else 0
            st.session_state.global_overtime = st.radio(
                "Heures supplémentaires",
                options=ot_options,
                index=ot_idx,
                horizontal=True,
                help="Filtrer selon la réalisation d'heures supplémentaires",
            )

            # Age filter
            min_age, max_age = int(df["Age"].min()), int(df["Age"].max())
            st.session_state.global_age_range = st.slider(
                "Tranche d'âge",
                min_value=min_age,
                max_value=max_age,
                value=st.session_state.global_age_range,
                help="Sélectionner l'intervalle d'âge des employés",
            )

            # Income filter
            min_inc, max_inc = int(df["MonthlyIncome"].min()), int(df["MonthlyIncome"].max())
            st.session_state.global_income_range = st.slider(
                "Revenu mensuel",
                min_value=min_inc,
                max_value=max_inc,
                value=st.session_state.global_income_range,
                step=500,
                help="Sélectionner l'intervalle de salaire mensuel",
            )

        filtered_df = filter_employee_data(
            filtered_df,
            tuple(st.session_state.global_departments),
            st.session_state.global_overtime,
            tuple(st.session_state.global_age_range),
            tuple(st.session_state.global_income_range),
        )

    render_sidebar_footer()
    return filtered_df


def render_sidebar_footer() -> None:
    """Render the sidebar footer with links and academic disclaimer."""
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <div style="font-size: 0.78rem; color: #6B7280; line-height: 1.4;">
            <p style="margin-bottom: 0.4rem;"><strong>📊 Data Mining 2026/2027</strong></p>
            <p style="margin-bottom: 0.4rem;">Mini-Projet — Classification supervisée</p>
            <p style="margin: 0;">⚠️ <em>Jeu de données synthétique à vocation académique.</em></p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Missing Artifact Warning ──────────────────────────────────────────
def show_missing_artifact_warning(artifact_name: str, fix_command: str = "") -> None:
    """Show a styled warning banner when an artifact is missing."""
    st.warning(
        f"⚠️ **Artefact manquant : {artifact_name}**\n\n"
        + (f"Pour générer cet élément, exécutez la commande suivante dans le terminal :\n```powershell\n{fix_command}\n```" if fix_command else "")
    )
