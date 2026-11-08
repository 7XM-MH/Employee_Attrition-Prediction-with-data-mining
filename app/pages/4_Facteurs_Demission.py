"""Page 4 — Attrition Factors: Feature importance and SHAP analysis."""
import streamlit as st
import sys
from pathlib import Path

# Ensure project root is on sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

st.set_page_config(
    page_title="Facteurs de Démission",
    page_icon="🔍",
    layout="wide",
)

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from app.utils import (
    inject_custom_css,
    render_sidebar_footer,
    load_data,
    load_model,
    load_json,
    show_missing_artifact_warning,
    MODEL_NAMES,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET,
    ID_COL,
    ALL_FEATURES,
)

inject_custom_css()
st.sidebar.title("🔍 Facteurs")
render_sidebar_footer()

# ── Color scheme ───────────────────────────────────────────────────────
COLOR_NO = "#1E88E5"
COLOR_YES = "#E53935"
COLOR_MAP = {"No": COLOR_NO, "Yes": COLOR_YES}

st.title("🔍 Facteurs de Démission")
st.markdown(
    "Identifiez les variables les plus influentes sur l'attrition des employés "
    "grâce à l'importance des features et aux analyses SHAP."
)
st.markdown("---")

# ── Load artifacts ─────────────────────────────────────────────────────
feature_importances = load_json("feature_importances.json")
df = load_data()


def display_name(key: str) -> str:
    """Get the French display name for a model key."""
    return MODEL_NAMES.get(key, key)


# ── Section 1: Feature importances ────────────────────────────────────
st.subheader("📊 Importance des Variables")

if feature_importances is not None:
    available_models = list(feature_importances.keys())
    if available_models:
        selected_model = st.selectbox(
            "Choisir un modèle",
            options=available_models,
            format_func=display_name,
        )

        fi = feature_importances[selected_model]
        fi_df = (
            pd.DataFrame(
                [{"Variable": k, "Importance": v} for k, v in fi.items()]
            )
            .sort_values("Importance", ascending=True)
            .reset_index(drop=True)
        )

        fig = px.bar(
            fi_df,
            x="Importance",
            y="Variable",
            orientation="h",
            title=f"Importance des Variables — {display_name(selected_model)}",
            color="Importance",
            color_continuous_scale="Blues",
        )
        fig.update_layout(
            template="plotly_white",
            height=max(400, len(fi_df) * 35),
            yaxis_title="",
            xaxis_title="Importance",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Show data table
        with st.expander("📋 Tableau des Importances"):
            display_fi = fi_df.sort_values("Importance", ascending=False).reset_index(drop=True)
            display_fi.index = display_fi.index + 1
            display_fi.index.name = "Rang"
            st.dataframe(
                display_fi.style.format({"Importance": "{:.4f}"}).bar(
                    subset=["Importance"], color="#90caf9"
                ),
                use_container_width=True,
            )
    else:
        st.info("ℹ️ Aucune importance de variable disponible.")
else:
    show_missing_artifact_warning("feature_importances.json", "python src/train.py")

st.markdown("---")

# ── Section 2: SHAP analysis ──────────────────────────────────────────
st.subheader("🔬 Analyse SHAP")

try:
    import shap
    import matplotlib.pyplot as plt

    model = load_model("best_model")

    if model is not None and df is not None:
        st.markdown(
            "Analyse SHAP sur un échantillon de 100 employés pour expliquer les décisions du meilleur modèle."
        )

        # Prepare sample data
        X = df.drop(columns=[TARGET, ID_COL], errors="ignore")
        sample = X.sample(n=min(100, len(X)), random_state=42)

        # Get pipeline steps
        pipeline = model
        preprocessor = pipeline[:-1]  # All steps except classifier
        classifier = pipeline.steps[-1][1]

        # Transform the sample
        X_transformed = preprocessor.transform(sample)

        # Get feature names
        try:
            feature_names_out = preprocessor.get_feature_names_out()
        except Exception:
            feature_names_out = None

        # Compute SHAP values
        with st.spinner("⏳ Calcul des valeurs SHAP en cours..."):
            explainer = shap.Explainer(classifier, feature_names=feature_names_out)
            shap_values = explainer(X_transformed)

        # For binary classification, select the positive class
        try:
            classes = list(model.classes_)
            yes_idx = classes.index("Yes") if "Yes" in classes else 1
        except Exception:
            yes_idx = 1

        if len(shap_values.shape) == 3:
            shap_values_display = shap_values[:, :, yes_idx]
        else:
            shap_values_display = shap_values

        # SHAP summary plot (beeswarm)
        col_shap1, col_shap2 = st.columns(2)

        with col_shap1:
            st.markdown("#### Diagramme en Essaim (Beeswarm)")
            fig_bee, ax_bee = plt.subplots(figsize=(10, 8))
            shap.plots.beeswarm(shap_values_display, max_display=15, show=False)
            plt.title("Impact des Variables sur la Prédiction", fontsize=14)
            plt.tight_layout()
            st.pyplot(fig_bee)
            plt.close(fig_bee)

        with col_shap2:
            st.markdown("#### Importance Moyenne (|SHAP|)")
            fig_bar_shap, ax_bar_shap = plt.subplots(figsize=(10, 8))
            shap.plots.bar(shap_values_display, max_display=15, show=False)
            plt.title("Importance Moyenne des Variables (SHAP)", fontsize=14)
            plt.tight_layout()
            st.pyplot(fig_bar_shap)
            plt.close(fig_bar_shap)

    elif model is None:
        show_missing_artifact_warning(
            "Modèle (best_model.joblib)",
            "python src/train.py",
        )
    else:
        show_missing_artifact_warning("Jeu de données")

except ImportError:
    st.info(
        "ℹ️ Le module **SHAP** n'est pas installé. "
        "Installez-le avec `pip install shap` pour voir l'analyse SHAP détaillée."
    )
except Exception as e:
    st.info(
        f"ℹ️ L'analyse SHAP n'a pas pu être générée : {e}\n\n"
        "Cela peut arriver avec certains types de modèles ou de données."
    )

st.markdown("---")

# ── Section 3: Bivariate analysis ─────────────────────────────────────
st.subheader("📈 Analyse Bivariée")

if df is not None:
    selected_feature = st.selectbox(
        "Choisir une variable à analyser",
        options=ALL_FEATURES,
        key="bivariate_feature",
    )

    if selected_feature in NUMERICAL_FEATURES:
        # Violin / Box plot for numerical features
        chart_type = st.radio(
            "Type de graphique",
            options=["Violin", "Boîte à Moustaches"],
            horizontal=True,
            key="chart_type_bivar",
        )
        if chart_type == "Violin":
            fig_biv = px.violin(
                df,
                x=TARGET,
                y=selected_feature,
                color=TARGET,
                color_discrete_map=COLOR_MAP,
                box=True,
                points="outliers",
                title=f"Distribution de {selected_feature} par Attrition",
            )
        else:
            fig_biv = px.box(
                df,
                x=TARGET,
                y=selected_feature,
                color=TARGET,
                color_discrete_map=COLOR_MAP,
                title=f"Distribution de {selected_feature} par Attrition",
            )
        fig_biv.update_layout(template="plotly_white", showlegend=False)
        st.plotly_chart(fig_biv, use_container_width=True)

    else:
        # Grouped bar chart for categorical features
        cross = (
            df.groupby([selected_feature, TARGET])
            .size()
            .reset_index(name="Nombre")
        )
        fig_biv = px.bar(
            cross,
            x=selected_feature,
            y="Nombre",
            color=TARGET,
            barmode="group",
            color_discrete_map=COLOR_MAP,
            title=f"Répartition de {selected_feature} par Attrition",
        )
        fig_biv.update_layout(template="plotly_white", legend_title_text="Attrition")
        st.plotly_chart(fig_biv, use_container_width=True)

        # Also show attrition rate by category
        rate = (
            df.groupby(selected_feature)[TARGET]
            .apply(lambda x: (x == "Yes").mean() * 100)
            .reset_index(name="Taux d'Attrition (%)")
        )
        fig_rate = px.bar(
            rate.sort_values("Taux d'Attrition (%)", ascending=True),
            x="Taux d'Attrition (%)",
            y=selected_feature,
            orientation="h",
            title=f"Taux d'Attrition par {selected_feature}",
            color="Taux d'Attrition (%)",
            color_continuous_scale=["#1E88E5", "#E53935"],
        )
        fig_rate.update_layout(template="plotly_white")
        st.plotly_chart(fig_rate, use_container_width=True)
else:
    show_missing_artifact_warning(
        "Jeu de données (Employee_Attrition_DataSet.csv)",
        "Placez le fichier dans data/raw/",
    )
