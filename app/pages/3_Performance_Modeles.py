"""Page 3 — Model Performance: Compare all trained models."""
import streamlit as st
import sys
from pathlib import Path

# Ensure project root is on sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

st.set_page_config(
    page_title="Performance des Modèles",
    page_icon="🏆",
    layout="wide",
)

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from app.utils import (
    inject_custom_css,
    render_sidebar_footer,
    load_json,
    show_missing_artifact_warning,
    MODEL_NAMES,
)

inject_custom_css()
st.sidebar.title("🏆 Performance")
render_sidebar_footer()

st.title("🏆 Performance des Modèles")
st.markdown(
    "Comparaison détaillée des performances de tous les modèles entraînés sur le jeu de données d'attrition."
)
st.markdown("---")

# ── Load all artifacts ─────────────────────────────────────────────────
metrics = load_json("metrics.json")
roc_data = load_json("roc_data.json")
confusion_matrices = load_json("confusion_matrices.json")
best_hyperparams = load_json("best_hyperparameters.json")
cv_results = load_json("cv_results_summary.json")


def display_name(key: str) -> str:
    """Get the French display name for a model key."""
    return MODEL_NAMES.get(key, key)


# ── Section 1: Metrics comparison table ───────────────────────────────
st.subheader("📊 Comparaison des Modèles")

if metrics is not None:
    rows = []
    for model_key, m in metrics.items():
        rows.append({
            "Modèle": display_name(model_key),
            "Accuracy": m.get("accuracy", 0),
            "Precision": m.get("precision", 0),
            "Recall": m.get("recall", 0),
            "F1-Score": m.get("f1", 0),
            "AUC-ROC": m.get("roc_auc", 0),
        })
    metrics_df = pd.DataFrame(rows).sort_values("F1-Score", ascending=False).reset_index(drop=True)

    # Highlight best values per column
    metric_cols = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]
    styled = metrics_df.style.format(
        {col: "{:.4f}" for col in metric_cols}
    ).highlight_max(
        subset=metric_cols, color="#c8e6c9"
    ).highlight_min(
        subset=metric_cols, color="#ffcdd2"
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # Bar chart comparison
    melted = metrics_df.melt(
        id_vars="Modèle", value_vars=metric_cols, var_name="Métrique", value_name="Valeur"
    )
    fig_bar = px.bar(
        melted,
        x="Modèle",
        y="Valeur",
        color="Métrique",
        barmode="group",
        title="Comparaison des Métriques par Modèle",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig_bar.update_layout(template="plotly_white", yaxis_range=[0, 1])
    st.plotly_chart(fig_bar, use_container_width=True)
else:
    show_missing_artifact_warning("metrics.json", "python src/train.py")

st.markdown("---")

# ── Section 2: ROC curves ─────────────────────────────────────────────
st.subheader("📈 Courbes ROC")

if roc_data is not None:
    fig_roc = go.Figure()

    colors = px.colors.qualitative.Set1
    for i, (model_key, data) in enumerate(roc_data.items()):
        fpr = data.get("fpr", [])
        tpr = data.get("tpr", [])
        auc_val = data.get("auc", 0)
        name = f"{display_name(model_key)} (AUC = {auc_val:.4f})"
        fig_roc.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=name,
                line=dict(color=colors[i % len(colors)], width=2),
            )
        )

    # Diagonal reference line
    fig_roc.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Référence (aléatoire)",
            line=dict(color="gray", width=1, dash="dash"),
        )
    )

    fig_roc.update_layout(
        title="Courbes ROC — Tous les Modèles",
        xaxis_title="Taux de Faux Positifs (FPR)",
        yaxis_title="Taux de Vrais Positifs (TPR)",
        template="plotly_white",
        legend=dict(x=0.5, y=0.02, font=dict(size=11)),
        height=550,
        xaxis=dict(constrain="domain"),
        yaxis=dict(scaleanchor="x", scaleratio=1),
    )
    st.plotly_chart(fig_roc, use_container_width=True)
else:
    show_missing_artifact_warning("roc_data.json", "python src/train.py")

st.markdown("---")

# ── Section 3: Confusion matrices ─────────────────────────────────────
st.subheader("🔲 Matrices de Confusion")

if confusion_matrices is not None:
    model_keys = list(confusion_matrices.keys())
    selected_models = st.multiselect(
        "Sélectionner les modèles à afficher",
        options=model_keys,
        default=model_keys[:3] if len(model_keys) >= 3 else model_keys,
        format_func=display_name,
    )

    if selected_models:
        # Display 2-3 per row
        n_cols = min(3, len(selected_models))
        for row_start in range(0, len(selected_models), n_cols):
            cols = st.columns(n_cols)
            for j, col in enumerate(cols):
                idx = row_start + j
                if idx < len(selected_models):
                    model_key = selected_models[idx]
                    cm = confusion_matrices[model_key]
                    with col:
                        labels = ["Non (réel)", "Oui (réel)"]
                        pred_labels = ["Non (prédit)", "Oui (prédit)"]
                        fig_cm = px.imshow(
                            cm,
                            text_auto=True,
                            color_continuous_scale="Blues",
                            x=pred_labels,
                            y=labels,
                            title=display_name(model_key),
                            aspect="auto",
                        )
                        fig_cm.update_layout(
                            height=350,
                            margin=dict(t=50, b=20, l=20, r=20),
                            template="plotly_white",
                        )
                        st.plotly_chart(fig_cm, use_container_width=True)
else:
    show_missing_artifact_warning("confusion_matrices.json", "python src/train.py")

st.markdown("---")

# ── Section 4: Best hyperparameters ───────────────────────────────────
st.subheader("⚙️ Meilleurs Hyperparamètres")

if best_hyperparams is not None:
    for model_key, params in best_hyperparams.items():
        with st.expander(f"🔧 {display_name(model_key)}", expanded=False):
            if params:
                params_df = pd.DataFrame(
                    [{"Paramètre": k, "Valeur": str(v)} for k, v in params.items()]
                )
                st.dataframe(params_df, use_container_width=True, hide_index=True)
            else:
                st.info("Aucun hyperparamètre à afficher pour ce modèle.")
else:
    show_missing_artifact_warning("best_hyperparameters.json", "python src/train.py")

st.markdown("---")

# ── Section 5: Cross-validation results ───────────────────────────────
st.subheader("📉 Résultats de la Validation Croisée")

if cv_results is not None:
    cv_rows = []
    for model_key, data in cv_results.items():
        cv_rows.append({
            "Modèle": display_name(model_key),
            "F1 Moyen (CV)": data.get("mean_cv_f1", 0),
            "Écart-type F1": data.get("std_cv_f1", 0),
        })
    cv_df = pd.DataFrame(cv_rows).sort_values("F1 Moyen (CV)", ascending=False).reset_index(drop=True)

    st.dataframe(
        cv_df.style.format({"F1 Moyen (CV)": "{:.4f}", "Écart-type F1": "{:.4f}"}),
        use_container_width=True,
        hide_index=True,
    )

    # Bar chart with error bars
    fig_cv = go.Figure()
    fig_cv.add_trace(
        go.Bar(
            x=cv_df["Modèle"],
            y=cv_df["F1 Moyen (CV)"],
            error_y=dict(
                type="data",
                array=cv_df["Écart-type F1"].tolist(),
                visible=True,
                color="#E53935",
                thickness=2,
                width=6,
            ),
            marker_color="#1E88E5",
            text=cv_df["F1 Moyen (CV)"].apply(lambda x: f"{x:.4f}"),
            textposition="outside",
        )
    )
    fig_cv.update_layout(
        title="Score F1 Moyen par Modèle (Validation Croisée)",
        xaxis_title="Modèle",
        yaxis_title="F1-Score Moyen",
        template="plotly_white",
        yaxis_range=[0, max(cv_df["F1 Moyen (CV)"].max() * 1.15, 1.0)],
        showlegend=False,
        height=450,
    )
    st.plotly_chart(fig_cv, use_container_width=True)
else:
    show_missing_artifact_warning("cv_results_summary.json", "python src/train.py")
