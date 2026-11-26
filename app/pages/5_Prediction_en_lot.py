"""Page 5 — Batch Prediction: Upload a CSV and predict attrition for multiple employees."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure project root is on sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

st.set_page_config(
    page_title="Prédiction en Lot — Attrition RH",
    page_icon="📦",
    layout="wide",
)

from app.utils import (
    load_css,
    load_bundle,
    get_plotly_template,
    kpi_card,
    risk_badge,
    section_header,
    render_sidebar,
    show_missing_artifact_warning,
    get_risk_level,
    ALL_FEATURES,
    PRIMARY_COLOR,
    SECONDARY_COLOR,
    COLOR_NO,
    COLOR_YES,
    COLOR_WARN,
    PLOTLY_TEMPLATE,
    PLOTLY_COLORWAY,
)

# Inject custom stylesheet
load_css()

# Render shared sidebar
render_sidebar(df=None, show_filters=False)
PLOTLY_TEMPLATE = get_plotly_template()

# ── Header ─────────────────────────────────────────────────────────────
section_header(
    "Prédiction en Lot & Traitement Massif",
    "Importez une liste de collaborateurs au format CSV pour calculer les scores d'attrition groupés",
    icon="📦",
)

# Template data for download and quick testing
template_data = {
    "EmployeeID": ["EMP1001", "EMP1002", "EMP1003", "EMP1004", "EMP1005", "EMP1006"],
    "Age": [35, 28, 48, 24, 41, 31],
    "Department": ["Engineering", "Sales", "Finance", "Marketing", "Operations", "IT"],
    "JobLevel": [3, 1, 4, 1, 3, 2],
    "YearsAtCompany": [8, 1, 15, 1, 10, 3],
    "MonthlyIncome": [12500, 4200, 18500, 3800, 9600, 7100],
    "JobSatisfaction": [4, 1, 3, 2, 4, 2],
    "WorkLifeBalance": [3, 1, 3, 2, 4, 1],
    "OverTime": ["No", "Yes", "No", "Yes", "No", "Yes"],
    "DistanceFromHome": [5.0, 28.0, 10.0, 32.0, 8.0, 22.0],
    "PromotionLast5Years": ["Yes", "No", "No", "No", "Yes", "No"],
    "PerformanceRating": [3, 2, 4, 2, 3, 3],
    "TrainingHoursLastYear": [22, 6, 18, 8, 25, 12],
}
template_df = pd.DataFrame(template_data)

# ── Top Download & Quick Test Bar ──────────────────────────────────────
col_left, col_right = st.columns([7, 5])

with col_left:
    st.markdown("#### 📄 Fichier CSV Modèle")
    st.caption("Le fichier importé doit contenir les 12 caractéristiques indispensables pour l'inférence du pipeline.")
    
    with st.expander("👁️ Consulter la structure des colonnes attendues", expanded=False):
        st.dataframe(template_df, use_container_width=True, hide_index=True)

    csv_template = template_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Télécharger la Trame CSV Type",
        data=csv_template,
        file_name="modele_prediction_attrition.csv",
        mime="text/csv",
    )

with col_right:
    st.markdown("#### 🚀 Test Rapide (Démo)")
    st.caption("Vous pouvez charger directement l'échantillon type de 6 collaborateurs sans téléverser de fichier.")
    use_sample = st.button("🧪 Charger l'échantillon de test immédiat", use_container_width=True)

st.markdown("---")

# ── File Upload Section ────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Téléversez votre fichier CSV de collaborateurs :",
    type=["csv"],
    help="Sélectionnez un fichier .csv encodé en UTF-8 comportant les colonnes requises.",
)

# Determine working dataframe
input_df = None

if uploaded_file is not None:
    try:
        input_df = pd.read_csv(uploaded_file)
    except Exception as err:
        st.error(f"❌ Erreur lors de la lecture du fichier CSV : {err}")
elif use_sample or ("batch_sample_loaded" in st.session_state and st.session_state.batch_sample_loaded):
    st.session_state.batch_sample_loaded = True
    input_df = template_df.copy()

# ── Processing & Predictions ───────────────────────────────────────────
if input_df is None:
    st.info(
        "📂 **Aucune donnée chargée pour le moment.**\n\n"
        "Veuillez glisser-déposer un fichier CSV ci-dessus, ou cliquer sur "
        "**« Charger l'échantillon de test immédiat »** pour tester la prédiction groupée."
    )
else:
    # Column validation
    missing_cols = [c for c in ALL_FEATURES if c not in input_df.columns]
    if missing_cols:
        st.error(
            f"❌ Colonnes obligatoires absentes dans le fichier : **{', '.join(missing_cols)}**\n\n"
            "Veuillez télécharger le modèle CSV ci-dessus pour vérifier la nomenclature exacte des champs."
        )
        st.stop()

    st.success(f"✅ Jeu de données valide — **{len(input_df):,} collaborateur(s)** détecté(s).")

    # Load Model bundle (probabilité Yes + seuil optimisé)
    bundle = load_bundle("best_model")
    if bundle is None:
        show_missing_artifact_warning(
            "Modèle entraîné (best_model.joblib)",
            "python scripts/train_pipeline.py",
        )
        st.stop()
    model = bundle["model"]
    threshold = float(bundle["threshold"])

    with st.spinner("⏳ Calcul des probabilités d'attrition sur l'ensemble de la cohorte..."):
        try:
            X_batch = input_df[ALL_FEATURES].copy()
            probas = model.predict_proba(X_batch)

            classes = list(getattr(model, "classes_", [0, 1]))
            yes_idx = classes.index("Yes") if "Yes" in classes else (classes.index(1) if 1 in classes else 1)
            p_yes = probas[:, yes_idx]
        except Exception as e:
            st.error(f"❌ Échec de l'inférence : {e}")
            st.stop()

    st.caption(
        f"Décision au seuil optimisé du bundle : **{threshold:.4f}**. "
        "Les paliers 30 % / 60 % ci-dessous sont des niveaux d'affichage, pas le seuil de décision."
    )

    # Enrich dataframe with scores & categories
    res_df = input_df.copy()
    res_df["Décision"] = ["Risque Départ" if p >= threshold else "Fidélisé" for p in p_yes]
    res_df["Probabilité Risque (%)"] = np.round(p_yes * 100, 2)
    res_df["Score Risque"] = p_yes
    res_df["Niveau Risque"] = [
        "🔴 Élevé" if p >= 0.6 else ("🟡 Modéré" if p >= 0.3 else "🟢 Faible")
        for p in p_yes
    ]

    # ── Summary KPIs ───────────────────────────────────────────────────
    st.markdown("#### 📊 Bilan Synthétique de la Cohorte")

    n_total = len(res_df)
    n_high = (p_yes >= 0.6).sum()
    n_med = ((p_yes >= 0.3) & (p_yes < 0.6)).sum()
    n_low = (p_yes < 0.3).sum()
    global_risk_pct = (n_high / n_total) * 100

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Effectif Évalué", f"{n_total:,}", border_color=PRIMARY_COLOR)
    with c2:
        kpi_card("Risque Élevé", f"{n_high:,}", delta=f"{global_risk_pct:.1f}% de la cohorte", delta_type="negative", border_color=COLOR_YES)
    with c3:
        kpi_card("Risque Modéré", f"{n_med:,}", delta="Vigilance", delta_type="neutral", border_color=COLOR_WARN)
    with c4:
        kpi_card("Risque Faible", f"{n_low:,}", delta="Stables", delta_type="positive", border_color=COLOR_NO)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Cohort Distribution Charts ─────────────────────────────────────
    chart_c1, chart_c2 = st.columns([5, 7])
    with chart_c1:
        risk_counts = pd.DataFrame({
            "Niveau": ["Faible (< 30%)", "Modéré (30-60%)", "Élevé (≥ 60%)"],
            "Nombre": [n_low, n_med, n_high],
        })
        fig_donut = px.pie(
            risk_counts,
            values="Nombre",
            names="Niveau",
            hole=0.45,
            color="Niveau",
            color_discrete_map={
                "Faible (< 30%)": COLOR_NO,
                "Modéré (30-60%)": COLOR_WARN,
                "Élevé (≥ 60%)": COLOR_YES,
            },
            title="Répartition des Risques",
            template=PLOTLY_TEMPLATE,
        )
        fig_donut.update_layout(margin=dict(t=40, b=20, l=20, r=20), height=320)
        st.plotly_chart(fig_donut, use_container_width=True)

    with chart_c2:
        if "Department" in res_df.columns:
            fig_bar = px.histogram(
                res_df,
                x="Department",
                color="Niveau Risque",
                barmode="stack",
                color_discrete_map={
                    "🟢 Faible": COLOR_NO,
                    "🟡 Modéré": COLOR_WARN,
                    "🔴 Élevé": COLOR_YES,
                },
                title="Distribution des Niveaux de Risque par Département",
                template=PLOTLY_TEMPLATE,
                height=320,
            )
            fig_bar.update_layout(margin=dict(t=40, b=20, l=20, r=20), xaxis_title="")
            st.plotly_chart(fig_bar, use_container_width=True)

    # ── Interactive Dataframe with Column Configuration ────────────────
    st.markdown("#### 📋 Résultats Détaillés & Export")

    filter_col1, filter_col2 = st.columns([4, 8])
    with filter_col1:
        selected_level = st.multiselect(
            "Filtrer par niveau de risque :",
            options=["🔴 Élevé", "🟡 Modéré", "🟢 Faible"],
            default=["🔴 Élevé", "🟡 Modéré", "🟢 Faible"],
        )
    
    display_df = res_df[res_df["Niveau Risque"].isin(selected_level)].sort_values("Score Risque", ascending=False)

    st.dataframe(
        display_df,
        column_config={
            "Probabilité Risque (%)": st.column_config.ProgressColumn(
                "Indice de Risque (%)",
                help="Probabilité estimée d'attrition en pourcentage",
                format="%.1f %%",
                min_value=0.0,
                max_value=100.0,
            ),
            "MonthlyIncome": st.column_config.NumberColumn(
                "Salaire (MAD)",
                format="%d MAD",
            ),
            "Probabilité Risque (%)": st.column_config.NumberColumn(
                "Probabilité",
                format="%.1f %%",
            ),
        },
        use_container_width=True,
        hide_index=True,
    )

    # Export Button — l'identifiant anonyme n'est pas ré-exporté (principe de moindre exposition).
    exported_df = display_df.drop(columns=[
        col for col in ("EmployeeID", "Attrition") if col in display_df.columns
    ])
    export_csv = exported_df.to_csv(index=False).encode("utf-8")
    st.caption("L'export exclut volontairement `EmployeeID` et `Attrition`.")
    st.download_button(
        label="📥 Exporter les Résultats Complets (CSV)",
        data=export_csv,
        file_name="predictions_attrition_lot.csv",
        mime="text/csv",
    )
