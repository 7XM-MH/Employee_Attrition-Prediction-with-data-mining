"""Page 1 — Dashboard EDA: Interactive exploration of the employee dataset."""
import streamlit as st
import sys
from pathlib import Path

# Ensure project root is on sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

st.set_page_config(
    page_title="Dashboard EDA",
    page_icon="📊",
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
    metric_card,
    show_missing_artifact_warning,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET,
    ID_COL,
)

inject_custom_css()

# ── Color scheme ───────────────────────────────────────────────────────
COLOR_NO = "#1E88E5"
COLOR_YES = "#E53935"
COLOR_MAP = {"No": COLOR_NO, "Yes": COLOR_YES}

# ── Load data ──────────────────────────────────────────────────────────
df = load_data()

if df is None:
    st.title("📊 Dashboard — Analyse Exploratoire")
    show_missing_artifact_warning(
        "Jeu de données (Employee_Attrition_DataSet.csv)",
        "Placez le fichier dans data/raw/",
    )
    render_sidebar_footer()
    st.stop()

# ── Sidebar filters ───────────────────────────────────────────────────
st.sidebar.title("🔎 Filtres")

departments = st.sidebar.multiselect(
    "Département",
    options=sorted(df["Department"].unique()),
    default=sorted(df["Department"].unique()),
)

overtime_filter = st.sidebar.radio(
    "Heures supplémentaires",
    options=["Tous", "Oui", "Non"],
    index=0,
)

promotion_filter = st.sidebar.radio(
    "Promotion (5 dernières années)",
    options=["Tous", "Oui", "Non"],
    index=0,
)

age_range = st.sidebar.slider(
    "Tranche d'âge",
    min_value=int(df["Age"].min()),
    max_value=int(df["Age"].max()),
    value=(int(df["Age"].min()), int(df["Age"].max())),
)

income_range = st.sidebar.slider(
    "Revenu mensuel",
    min_value=int(df["MonthlyIncome"].min()),
    max_value=int(df["MonthlyIncome"].max()),
    value=(int(df["MonthlyIncome"].min()), int(df["MonthlyIncome"].max())),
)

years_range = st.sidebar.slider(
    "Ancienneté (années)",
    min_value=int(df["YearsAtCompany"].min()),
    max_value=int(df["YearsAtCompany"].max()),
    value=(int(df["YearsAtCompany"].min()), int(df["YearsAtCompany"].max())),
)

render_sidebar_footer()

# ── Apply filters ──────────────────────────────────────────────────────
filtered = df.copy()
filtered = filtered[filtered["Department"].isin(departments)]

if overtime_filter == "Oui":
    filtered = filtered[filtered["OverTime"] == "Yes"]
elif overtime_filter == "Non":
    filtered = filtered[filtered["OverTime"] == "No"]

if promotion_filter == "Oui":
    filtered = filtered[filtered["PromotionLast5Years"] == "Yes"]
elif promotion_filter == "Non":
    filtered = filtered[filtered["PromotionLast5Years"] == "No"]

filtered = filtered[
    (filtered["Age"] >= age_range[0])
    & (filtered["Age"] <= age_range[1])
    & (filtered["MonthlyIncome"] >= income_range[0])
    & (filtered["MonthlyIncome"] <= income_range[1])
    & (filtered["YearsAtCompany"] >= years_range[0])
    & (filtered["YearsAtCompany"] <= years_range[1])
]

# ── Header ─────────────────────────────────────────────────────────────
st.title("📊 Dashboard — Analyse Exploratoire")

# Filtered count summary
col_info1, col_info2, col_info3, col_info4 = st.columns(4)
with col_info1:
    metric_card("Employés Affichés", f"{len(filtered):,}", "metric-card-blue")
with col_info2:
    metric_card("Total Employés", f"{len(df):,}", "metric-card-purple")
with col_info3:
    attrition_pct = (
        (filtered["Attrition"] == "Yes").sum() / len(filtered) * 100
        if len(filtered) > 0
        else 0
    )
    metric_card("Taux d'Attrition (filtré)", f"{attrition_pct:.1f}%", "metric-card-red")
with col_info4:
    avg_income = filtered["MonthlyIncome"].mean() if len(filtered) > 0 else 0
    metric_card("Salaire Moyen (filtré)", f"{avg_income:,.0f}", "metric-card-green")

st.markdown("<br>", unsafe_allow_html=True)

if len(filtered) == 0:
    st.warning("⚠️ Aucun employé ne correspond aux filtres sélectionnés.")
    st.stop()

# ── Tabs ───────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["📈 Distribution", "📊 Catégoriel", "🔗 Corrélation", "📉 Attrition", "📋 Statistiques", "🔍 Données"]
)

# ── Tab 1: Distribution ───────────────────────────────────────────────
with tab1:
    st.subheader("📈 Distribution des Variables Numériques")
    selected_num = st.selectbox(
        "Choisir une variable numérique",
        options=NUMERICAL_FEATURES,
        key="dist_num",
    )
    fig = px.histogram(
        filtered,
        x=selected_num,
        color=TARGET,
        barmode="overlay",
        color_discrete_map=COLOR_MAP,
        nbins=30,
        title=f"Distribution de {selected_num} par Attrition",
        labels={selected_num: selected_num, "count": "Nombre d'employés"},
        opacity=0.7,
    )
    fig.update_layout(
        bargap=0.05,
        legend_title_text="Attrition",
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Show all numerical distributions in a grid
    st.markdown("#### Toutes les distributions numériques")
    cols = st.columns(3)
    for i, feat in enumerate(NUMERICAL_FEATURES):
        with cols[i % 3]:
            fig_small = px.histogram(
                filtered,
                x=feat,
                color=TARGET,
                barmode="overlay",
                color_discrete_map=COLOR_MAP,
                nbins=25,
                title=feat,
                opacity=0.7,
                height=300,
            )
            fig_small.update_layout(
                showlegend=False,
                template="plotly_white",
                margin=dict(t=40, b=20, l=20, r=20),
            )
            st.plotly_chart(fig_small, use_container_width=True)

# ── Tab 2: Categorical ────────────────────────────────────────────────
with tab2:
    st.subheader("📊 Variables Catégorielles")
    for cat_feat in CATEGORICAL_FEATURES:
        cross = (
            filtered.groupby([cat_feat, TARGET])
            .size()
            .reset_index(name="Nombre")
        )
        fig = px.bar(
            cross,
            x=cat_feat,
            y="Nombre",
            color=TARGET,
            barmode="group",
            color_discrete_map=COLOR_MAP,
            title=f"Répartition de {cat_feat} par Attrition",
            labels={"Nombre": "Nombre d'employés"},
        )
        fig.update_layout(template="plotly_white", legend_title_text="Attrition")
        st.plotly_chart(fig, use_container_width=True)

# ── Tab 3: Correlation ────────────────────────────────────────────────
with tab3:
    st.subheader("🔗 Matrice de Corrélation")
    numeric_df = filtered[NUMERICAL_FEATURES]
    corr_matrix = numeric_df.corr()
    fig = px.imshow(
        corr_matrix,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Corrélation entre les Variables Numériques",
        aspect="auto",
    )
    fig.update_layout(template="plotly_white", height=600)
    st.plotly_chart(fig, use_container_width=True)

# ── Tab 4: Attrition analysis ─────────────────────────────────────────
with tab4:
    st.subheader("📉 Analyse de l'Attrition")

    # Attrition by department
    dept_attrition = (
        filtered.groupby("Department")[TARGET]
        .apply(lambda x: (x == "Yes").mean() * 100)
        .reset_index(name="Taux d'Attrition (%)")
    )
    fig1 = px.bar(
        dept_attrition.sort_values("Taux d'Attrition (%)", ascending=True),
        x="Taux d'Attrition (%)",
        y="Department",
        orientation="h",
        title="Taux d'Attrition par Département",
        color="Taux d'Attrition (%)",
        color_continuous_scale=["#1E88E5", "#E53935"],
    )
    fig1.update_layout(template="plotly_white", yaxis_title="Département")
    st.plotly_chart(fig1, use_container_width=True)

    col_a, col_b = st.columns(2)

    # Attrition by JobLevel
    with col_a:
        level_attrition = (
            filtered.groupby("JobLevel")[TARGET]
            .apply(lambda x: (x == "Yes").mean() * 100)
            .reset_index(name="Taux d'Attrition (%)")
        )
        fig2 = px.bar(
            level_attrition,
            x="JobLevel",
            y="Taux d'Attrition (%)",
            title="Taux d'Attrition par Niveau de Poste",
            color="Taux d'Attrition (%)",
            color_continuous_scale=["#1E88E5", "#E53935"],
        )
        fig2.update_layout(template="plotly_white", xaxis_title="Niveau de Poste")
        st.plotly_chart(fig2, use_container_width=True)

    # Attrition by OverTime
    with col_b:
        ot_attrition = (
            filtered.groupby("OverTime")[TARGET]
            .apply(lambda x: (x == "Yes").mean() * 100)
            .reset_index(name="Taux d'Attrition (%)")
        )
        fig3 = px.bar(
            ot_attrition,
            x="OverTime",
            y="Taux d'Attrition (%)",
            title="Taux d'Attrition par Heures Supplémentaires",
            color="OverTime",
            color_discrete_map={"Yes": COLOR_YES, "No": COLOR_NO},
        )
        fig3.update_layout(template="plotly_white", xaxis_title="Heures Supplémentaires")
        st.plotly_chart(fig3, use_container_width=True)

    # Attrition by years bucket
    filtered_copy = filtered.copy()
    bins = [0, 2, 5, 10, 15, 20, 36]
    labels_years = ["0-2", "3-5", "6-10", "11-15", "16-20", "21+"]
    filtered_copy["Ancienneté"] = pd.cut(
        filtered_copy["YearsAtCompany"], bins=bins, labels=labels_years, right=True
    )
    years_attrition = (
        filtered_copy.groupby("Ancienneté", observed=False)[TARGET]
        .apply(lambda x: (x == "Yes").mean() * 100)
        .reset_index(name="Taux d'Attrition (%)")
    )
    fig4 = px.bar(
        years_attrition,
        x="Ancienneté",
        y="Taux d'Attrition (%)",
        title="Taux d'Attrition par Tranche d'Ancienneté",
        color="Taux d'Attrition (%)",
        color_continuous_scale=["#1E88E5", "#E53935"],
    )
    fig4.update_layout(template="plotly_white", xaxis_title="Ancienneté (années)")
    st.plotly_chart(fig4, use_container_width=True)

# ── Tab 5: Descriptive stats ──────────────────────────────────────────
with tab5:
    st.subheader("📋 Statistiques Descriptives")
    desc = filtered[NUMERICAL_FEATURES].describe().T
    desc.columns = [
        "Nombre", "Moyenne", "Écart-type", "Min",
        "Q1 (25%)", "Médiane (50%)", "Q3 (75%)", "Max",
    ]
    st.dataframe(
        desc.style.format("{:.2f}").background_gradient(cmap="Blues", subset=["Moyenne"]),
        use_container_width=True,
    )

    st.markdown("#### Répartition de la variable cible")
    target_counts = filtered[TARGET].value_counts().reset_index()
    target_counts.columns = ["Attrition", "Nombre"]
    fig_target = px.pie(
        target_counts,
        names="Attrition",
        values="Nombre",
        color="Attrition",
        color_discrete_map=COLOR_MAP,
        title="Répartition de l'Attrition",
        hole=0.4,
    )
    fig_target.update_layout(template="plotly_white")
    st.plotly_chart(fig_target, use_container_width=True)

# ── Tab 6: Raw data ───────────────────────────────────────────────────
with tab6:
    st.subheader("🔍 Données Filtrées")
    search_term = st.text_input("🔎 Rechercher dans les données", "")
    display_df = filtered.copy()
    if search_term:
        mask = display_df.astype(str).apply(
            lambda row: row.str.contains(search_term, case=False).any(), axis=1
        )
        display_df = display_df[mask]

    st.dataframe(display_df, use_container_width=True, height=500)
    st.caption(f"Affichage de {len(display_df):,} lignes sur {len(filtered):,} filtrées ({len(df):,} au total)")

    # Download button
    csv_data = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Télécharger les données filtrées (CSV)",
        data=csv_data,
        file_name="donnees_filtrees.csv",
        mime="text/csv",
    )
