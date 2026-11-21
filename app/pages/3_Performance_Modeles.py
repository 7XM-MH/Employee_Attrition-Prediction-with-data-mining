"""Page 3 — Model Performance: Compare all trained models."""
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
    page_title="Performance des Modèles — Attrition RH",
    page_icon="🏆",
    layout="wide",
)

from app.utils import (
    load_css,
    load_json,
    get_plotly_template,
    kpi_card,
    section_header,
    render_sidebar,
    show_missing_artifact_warning,
    MODEL_NAMES,
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
    "Benchmark & Performance des Modèles de Classification",
    "Évaluation comparative des 6 algorithmes sur le jeu de test indépendant (20% stratifié)",
    icon="🏆",
)

# ── Load all artifacts ─────────────────────────────────────────────────
metrics = load_json("metrics.json")
roc_data = load_json("roc_data.json")
confusion_matrices = load_json("confusion_matrices.json")
best_hyperparams = load_json("best_hyperparameters.json")
cv_results = load_json("cv_results_summary.json")


def display_name(key: str) -> str:
    """Get the French display name for a model key."""
    return MODEL_NAMES.get(key, key)


if metrics is None:
    st.info(
        "ℹ️ **Les artefacts d'entraînement ne sont pas encore générés.**\n\n"
        "Pour entraîner les 6 algorithmes (Zero-R, KNN, Decision Tree, Random Forest, Naive Bayes, SVM) "
        "et générer les métriques et courbes ROC, lancez dans votre terminal :\n\n"
        "```powershell\npython scripts/train_pipeline.py\n```"
    )

    st.markdown("##### 📋 Rappel des 6 modèles évalués dans ce projet")
    st.markdown(
        """
        | Algorithme | Type | Particularité |
        |---|---|---|
        | **Zero-R (Baseline)** | Règle majoritaire | Référence minimale (prédit toujours la classe majoritaire) |
        | **K-Plus Proches Voisins (K-NN)** | Non paramétrique | Distance Euclidienne/Manhattan normalisée |
        | **Arbre de Décision (CART)** | Basé sur les règles | Gini / Entropie avec élagage (max_depth) |
        | **Forêt Aléatoire (Random Forest)** | Ensembliste (Bagging) | Agrégation d'arbres avec RandomizedSearch |
        | **Naïve Bayes Gaussien** | Probabiliste | Hypothèse d'indépendance conditionnelle |
        | **Support Vector Machine (SVM)** | Séparateur à vaste marge | Noyaux RBF et Linéaire avec probabilités Platt |
        """
    )
else:
    # ── Section 1: Overview & Leaderboard ─────────────────────────────
    rows = []
    for model_key, m in metrics.items():
        rows.append({
            "key": model_key,
            "Modèle": display_name(model_key),
            "Accuracy": m.get("accuracy", 0),
            "Precision": m.get("precision", 0),
            "Recall": m.get("recall", 0),
            "F1-Score": m.get("f1", 0),
            "AUC-ROC": m.get("roc_auc") if m.get("roc_auc") is not None else 0.5,
        })
    metrics_df = pd.DataFrame(rows).sort_values("F1-Score", ascending=False).reset_index(drop=True)
    best_row = metrics_df.iloc[0]

    st.markdown("#### 🥇 Modèle Lauréat")
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi_card("Meilleur Modèle", best_row["Modèle"], border_color=PRIMARY_COLOR, help_text="Modèle avec le meilleur score F1 sur le jeu de test")
    with k2:
        kpi_card("F1-Score (Test)", f"{best_row['F1-Score']:.4f}", border_color=COLOR_NO, help_text="Moyenne harmonique Precision-Recall")
    with k3:
        kpi_card("AUC-ROC", f"{best_row['AUC-ROC']:.4f}", border_color=SECONDARY_COLOR, help_text="Aire sous la courbe ROC")
    with k4:
        kpi_card("Accuracy Globale", f"{best_row['Accuracy']*100:.2f}%", border_color=COLOR_WARN, help_text="Taux global de bonnes prédictions")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Section 2: Metrics Table & Bar Chart ──────────────────────────
    selected_section = st.radio(
        "Section de performance",
        ["Métriques", "Courbes ROC", "Matrices de confusion", "Validation croisée", "Hyperparamètres"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if selected_section == "Métriques":
        st.markdown("##### Tableau comparatif complet")
        metric_cols = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]
        styled = metrics_df[["Modèle"] + metric_cols].style.format(
            {col: "{:.4f}" for col in metric_cols}
        ).highlight_max(
            subset=metric_cols, color="rgba(16, 185, 129, 0.25)"
        ).highlight_min(
            subset=metric_cols, color="rgba(239, 68, 68, 0.2)"
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)

        st.markdown("##### Visualisation graphique des métriques")
        melted = metrics_df.melt(
            id_vars="Modèle", value_vars=metric_cols, var_name="Métrique", value_name="Valeur"
        )
        fig_bar = px.bar(
            melted,
            x="Modèle",
            y="Valeur",
            color="Métrique",
            barmode="group",
            title="Comparaison multi-métriques par algorithme",
            template=PLOTLY_TEMPLATE,
            color_discrete_sequence=PLOTLY_COLORWAY,
            height=400,
        )
        fig_bar.update_layout(yaxis_range=[0, 1.05], legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_bar, use_container_width=True)

    if selected_section == "Courbes ROC":
        st.markdown("##### Courbes ROC (Receiver Operating Characteristic)")
        if roc_data is not None:
            fig_roc = go.Figure()
            for i, (m_key, data) in enumerate(roc_data.items()):
                fpr = data.get("fpr", [])
                tpr = data.get("tpr", [])
                auc_val = data.get("auc", 0)
                auc_str = f"{auc_val:.4f}" if auc_val is not None else "N/A"
                fig_roc.add_trace(
                    go.Scatter(
                        x=fpr,
                        y=tpr,
                        mode="lines",
                        name=f"{display_name(m_key)} (AUC = {auc_str})",
                        line=dict(color=PLOTLY_COLORWAY[i % len(PLOTLY_COLORWAY)], width=2.5),
                    )
                )

            # Diagonal baseline
            fig_roc.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode="lines",
                    name="Aléatoire (AUC = 0.5000)",
                    line=dict(dash="dash", color="#9CA3AF", width=1.5),
                )
            )

            fig_roc.update_layout(
                title="Courbes ROC comparatives",
                xaxis_title="Taux de Faux Positifs (1 - Spécificité)",
                yaxis_title="Taux de Vrais Positifs (Sensibilité / Recall)",
                template=PLOTLY_TEMPLATE,
                height=480,
                legend=dict(yanchor="bottom", y=0.05, xanchor="right", x=0.98),
            )
            st.plotly_chart(fig_roc, use_container_width=True)
        else:
            show_missing_artifact_warning("roc_data.json")

    if selected_section == "Matrices de confusion":
        st.markdown("##### Matrices de Confusion sur le Jeu de Test")
        if confusion_matrices is not None:
            selected_m = st.selectbox(
                "Sélectionner un modèle pour afficher sa matrice de confusion :",
                options=list(confusion_matrices.keys()),
                format_func=display_name,
                key="cm_select",
            )
            cm_vals = confusion_matrices[selected_m]
            cm_arr = np.array(cm_vals)

            fig_cm = px.imshow(
                cm_arr,
                labels=dict(x="Classe Prédite", y="Classe Réelle", color="Effectif"),
                x=["Resté (No)", "Parti (Yes)"],
                y=["Resté (No)", "Parti (Yes)"],
                text_auto=True,
                color_continuous_scale="Blues",
                title=f"Matrice de Confusion — {display_name(selected_m)}",
                template=PLOTLY_TEMPLATE,
                height=380,
            )
            st.plotly_chart(fig_cm, use_container_width=True)

            # Calculations
            tn, fp = cm_arr[0, 0], cm_arr[0, 1]
            fn, tp = cm_arr[1, 0], cm_arr[1, 1]
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                kpi_card("Vrais Négatifs (VN)", f"{tn:,}", border_color=COLOR_NO)
            with c2:
                kpi_card("Faux Positifs (FP)", f"{fp:,}", border_color=COLOR_WARN)
            with c3:
                kpi_card("Faux Négatifs (FN)", f"{fn:,}", border_color=COLOR_YES)
            with c4:
                kpi_card("Vrais Positifs (VP)", f"{tp:,}", border_color=PRIMARY_COLOR)
        else:
            show_missing_artifact_warning("confusion_matrices.json")

    if selected_section == "Validation croisée":
        st.markdown("##### Stabilité par Validation Croisée Stratifiée (5-Fold)")
        if cv_results is not None:
            cv_rows = []
            for k, val in cv_results.items():
                m_score = val.get("mean_cv_f1")
                s_score = val.get("std_cv_f1")
                cv_rows.append({
                    "Modèle": display_name(k),
                    "F1 Moyen (CV)": f"{m_score:.4f}" if m_score is not None else "N/A",
                    "Écart-Type (±σ)": f"± {s_score:.4f}" if s_score is not None else "N/A",
                })
            cv_df = pd.DataFrame(cv_rows)
            st.dataframe(cv_df, use_container_width=True, hide_index=True)
        else:
            show_missing_artifact_warning("cv_results_summary.json")

    if selected_section == "Hyperparamètres":
        st.markdown("##### Hyperparamètres Optimaux Retenus (GridSearch / RandomSearch)")
        if best_hyperparams is not None:
            param_rows = []
            for k, pdict in best_hyperparams.items():
                p_str = ", ".join([f"{k_p}={v_p}" for k_p, v_p in pdict.items()]) if pdict else "Aucun (Baseline)"
                param_rows.append({
                    "Modèle": display_name(k),
                    "Configuration Optimale": p_str,
                })
            param_df = pd.DataFrame(param_rows)
            st.dataframe(param_df, use_container_width=True, hide_index=True)
        else:
            show_missing_artifact_warning("best_hyperparameters.json")
