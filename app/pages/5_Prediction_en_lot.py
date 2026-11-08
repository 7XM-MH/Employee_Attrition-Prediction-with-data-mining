"""Page 5 — Batch Prediction: Upload a CSV and predict attrition for multiple employees."""
import streamlit as st
import sys
from pathlib import Path

# Ensure project root is on sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

st.set_page_config(
    page_title="Prédiction en Lot",
    page_icon="📦",
    layout="wide",
)

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from app.utils import (
    inject_custom_css,
    render_sidebar_footer,
    load_model,
    metric_card,
    show_missing_artifact_warning,
    get_risk_level,
    get_risk_color,
    ALL_FEATURES,
    DEPARTMENT_LIST,
)

inject_custom_css()
st.sidebar.title("📦 Prédiction en Lot")
render_sidebar_footer()

st.title("📦 Prédiction en Lot")
st.markdown(
    "Chargez un fichier CSV contenant les données de plusieurs employés "
    "pour obtenir des prédictions groupées de risque d'attrition."
)
st.markdown("---")

# ── Template download ──────────────────────────────────────────────────
st.subheader("📄 Format Attendu")
st.markdown(
    "Le fichier CSV doit contenir les colonnes suivantes. "
    "Vous pouvez télécharger un modèle vierge ci-dessous."
)

# Generate template
template_data = {
    "Age": [35, 28],
    "Department": ["Engineering", "Sales"],
    "JobLevel": [3, 1],
    "YearsAtCompany": [8, 1],
    "MonthlyIncome": [12000, 4500],
    "JobSatisfaction": [4, 2],
    "WorkLifeBalance": [3, 2],
    "OverTime": ["No", "Yes"],
    "DistanceFromHome": [5.0, 30.0],
    "PromotionLast5Years": ["Yes", "No"],
    "PerformanceRating": [3, 2],
    "TrainingHoursLastYear": [20, 5],
}
template_df = pd.DataFrame(template_data)

st.dataframe(template_df, use_container_width=True, hide_index=True)

template_csv = template_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 Télécharger le Modèle CSV",
    data=template_csv,
    file_name="modele_prediction_attrition.csv",
    mime="text/csv",
)

st.markdown("---")

# ── File uploader ──────────────────────────────────────────────────────
st.subheader("📤 Charger le Fichier")
uploaded_file = st.file_uploader(
    "Glissez-déposez ou sélectionnez un fichier CSV",
    type=["csv"],
    help="Le fichier doit contenir toutes les colonnes listées ci-dessus.",
)

if uploaded_file is not None:
    # Read uploaded file
    try:
        input_df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"❌ Erreur lors de la lecture du fichier : {e}")
        st.stop()

    # Validate columns
    missing_cols = [col for col in ALL_FEATURES if col not in input_df.columns]
    if missing_cols:
        st.error(
            f"❌ Colonnes manquantes dans le fichier : **{', '.join(missing_cols)}**\n\n"
            "Veuillez vérifier que votre fichier contient toutes les colonnes requises."
        )
        st.stop()

    st.success(f"✅ Fichier chargé avec succès — {len(input_df):,} employé(s) détecté(s).")

    # Show preview
    with st.expander("👁️ Aperçu des données chargées", expanded=True):
        st.dataframe(input_df.head(10), use_container_width=True, hide_index=True)

    st.markdown("---")

    # Load model
    model = load_model("best_model")

    if model is None:
        show_missing_artifact_warning(
            "Modèle (best_model.joblib)",
            "python src/train.py",
        )
        st.stop()

    # Predict
    with st.spinner("⏳ Prédiction en cours..."):
        try:
            # Prepare input — keep only the required features
            X = input_df[ALL_FEATURES].copy()
            predictions = model.predict(X)
            probas = model.predict_proba(X)

            # Find index for "Yes" class
            classes = list(model.classes_)
            yes_idx = classes.index("Yes") if "Yes" in classes else 1

            attrition_probas = probas[:, yes_idx]
        except Exception as e:
            st.error(f"❌ Erreur lors de la prédiction : {e}")
            st.stop()

    # Build results dataframe
    results = input_df.copy()
    results["Prédiction"] = predictions
    results["Probabilité (%)"] = np.round(attrition_probas * 100, 2)
    results["Niveau de Risque"] = [get_risk_level(p)[0] for p in attrition_probas]

    # ── Results display ────────────────────────────────────────────────
    st.subheader("📊 Résultats des Prédictions")

    # Summary KPIs
    total = len(results)
    n_low = (results["Niveau de Risque"] == "Faible").sum()
    n_medium = (results["Niveau de Risque"] == "Modéré").sum()
    n_high = (results["Niveau de Risque"] == "Élevé").sum()
    pct_attrition = (results["Prédiction"] == "Yes").mean() * 100

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    with kpi1:
        metric_card("Total Employés", f"{total:,}", "metric-card-blue")
    with kpi2:
        metric_card("🟢 Risque Faible", f"{n_low:,}", "metric-card-green")
    with kpi3:
        metric_card("🟡 Risque Modéré", f"{n_medium:,}", "metric-card-orange")
    with kpi4:
        metric_card("🔴 Risque Élevé", f"{n_high:,}", "metric-card-red")
    with kpi5:
        metric_card("Taux d'Attrition Prédit", f"{pct_attrition:.1f}%", "metric-card-purple")

    st.markdown("<br>", unsafe_allow_html=True)

    # Risk level distribution chart
    risk_counts = pd.DataFrame({
        "Niveau de Risque": ["🟢 Faible", "🟡 Modéré", "🔴 Élevé"],
        "Nombre": [n_low, n_medium, n_high],
    })
    fig_risk = px.bar(
        risk_counts,
        x="Niveau de Risque",
        y="Nombre",
        color="Niveau de Risque",
        color_discrete_map={
            "🟢 Faible": "#28a745",
            "🟡 Modéré": "#ffc107",
            "🔴 Élevé": "#dc3545",
        },
        title="Répartition par Niveau de Risque",
        text="Nombre",
    )
    fig_risk.update_layout(
        template="plotly_white",
        showlegend=False,
        xaxis_title="",
        yaxis_title="Nombre d'employés",
    )
    fig_risk.update_traces(textposition="outside")
    st.plotly_chart(fig_risk, use_container_width=True)

    # Probability distribution histogram
    fig_prob = px.histogram(
        results,
        x="Probabilité (%)",
        nbins=20,
        title="Distribution des Probabilités d'Attrition",
        color_discrete_sequence=["#1E88E5"],
    )
    fig_prob.update_layout(
        template="plotly_white",
        xaxis_title="Probabilité d'Attrition (%)",
        yaxis_title="Nombre d'employés",
    )
    st.plotly_chart(fig_prob, use_container_width=True)

    # Full results table with color coding
    st.subheader("📋 Tableau des Résultats")

    def highlight_risk(row):
        """Apply background color based on risk level."""
        risk = row["Niveau de Risque"]
        if risk == "Élevé":
            return ["background-color: #f8d7da"] * len(row)
        elif risk == "Modéré":
            return ["background-color: #fff3cd"] * len(row)
        else:
            return ["background-color: #d4edda"] * len(row)

    styled_results = results.style.apply(highlight_risk, axis=1).format(
        {"Probabilité (%)": "{:.2f}"}
    )
    st.dataframe(styled_results, use_container_width=True, height=500, hide_index=True)

    # Download results
    results_csv = results.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Télécharger les Résultats (CSV)",
        data=results_csv,
        file_name="resultats_prediction_attrition.csv",
        mime="text/csv",
        type="primary",
    )
else:
    st.info(
        "👆 Chargez un fichier CSV ci-dessus pour commencer la prédiction en lot.\n\n"
        "Vous pouvez utiliser le modèle CSV fourni comme point de départ."
    )
