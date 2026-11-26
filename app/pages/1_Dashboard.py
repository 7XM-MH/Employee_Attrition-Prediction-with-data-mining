"""Page 1 — Dashboard EDA: Interactive exploration of the employee dataset."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

# Ensure project root is on sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

st.set_page_config(
    page_title="Dashboard EDA — Attrition RH",
    page_icon="📊",
    layout="wide",
)

from app.utils import (
    load_css,
    load_data,
    get_plotly_template,
    kpi_card,
    section_header,
    render_sidebar,
    show_missing_artifact_warning,
    PRIMARY_COLOR,
    SECONDARY_COLOR,
    COLOR_NO,
    COLOR_YES,
    COLOR_WARN,
    COLOR_MAP,
    PLOTLY_TEMPLATE,
    PLOTLY_COLORWAY,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET,
    ID_COL,
)

# Inject custom stylesheet
load_css()

# Load raw data
raw_df = load_data()

if raw_df is None:
    st.title("📊 Dashboard — Analyse Exploratoire")
    show_missing_artifact_warning(
        "Jeu de données (Employee_Attrition_DataSet.csv)",
        "Placez le fichier dans data/raw/",
    )
    render_sidebar(df=None, show_filters=False)
    st.stop()

# Render sidebar with active filters synced in session_state
filtered = render_sidebar(df=raw_df, show_filters=True)
PLOTLY_TEMPLATE = get_plotly_template()

# ── Header & KPI summary ───────────────────────────────────────────────
section_header(
    "Tableau de Bord Exploratoire (EDA)",
    "Analysez les facteurs d'attrition sur la population filtrée en temps réel",
    icon="📊",
)

col_info1, col_info2, col_info3, col_info4 = st.columns(4)

total_filtered = len(filtered)
total_raw = len(raw_df)
attrition_count = (filtered["Attrition"] == "Yes").sum() if total_filtered > 0 else 0
attrition_pct = (attrition_count / total_filtered * 100) if total_filtered > 0 else 0.0
avg_income = filtered["MonthlyIncome"].mean() if total_filtered > 0 else 0.0

with col_info1:
    kpi_card(
        "Employés Filtrés",
        f"{total_filtered:,}",
        delta=f"{(total_filtered/total_raw)*100:.1f}% de la base",
        border_color=PRIMARY_COLOR,
        help_text="Nombre de collaborateurs correspondant aux critères de la barre latérale",
    )
with col_info2:
    kpi_card(
        "Base Globale",
        f"{total_raw:,}",
        delta="Total entreprise",
        border_color=SECONDARY_COLOR,
        help_text="Volume total d'enregistrements du jeu de données",
    )
with col_info3:
    kpi_card(
        "Taux d'Attrition Filtré",
        f"{attrition_pct:.1f}%",
        delta=f"{attrition_count} départs",
        delta_type="negative",
        border_color=COLOR_YES,
        help_text="Proportion des départs volontaires dans l'échantillon sélectionné",
    )
with col_info4:
    kpi_card(
        "Salaire Moyen Filtré",
        f"{avg_income:,.0f} MAD",
        delta="Revenu mensuel",
        border_color=COLOR_NO,
        help_text="Moyenne des rémunérations pour la sélection courante",
    )

st.markdown("<br>", unsafe_allow_html=True)

if total_filtered == 0:
    st.warning("⚠️ Aucun employé ne correspond aux filtres sélectionnés dans la barre latérale. Ajustez les critères pour afficher les graphiques.")
    st.stop()

# ── Tabs ───────────────────────────────────────────────────────────────
selected_section = st.radio(
    "Section d'analyse",
    ["Contrôle qualité", "Distributions", "Facteurs catégoriels", "Corrélations", "Taux d'attrition", "Statistiques", "Tests statistiques", "Données filtrées"],
    horizontal=True,
    label_visibility="collapsed",
)

# ── Contrôle qualité des données ────────────────────────────────────────
if selected_section == "Contrôle qualité":
    st.markdown("#### 🔎 Audit qualité de la base brute")

    duplicate_rows = int(raw_df.duplicated().sum())
    duplicate_ids = int(raw_df[ID_COL].duplicated().sum()) if ID_COL in raw_df.columns else 0
    missing = raw_df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    q1, q2, q3, q4 = st.columns(4)
    with q1:
        kpi_card("Lignes", f"{len(raw_df):,}", border_color=PRIMARY_COLOR)
    with q2:
        kpi_card("Doublons complets", f"{duplicate_rows:,}", delta="Lignes identiques", border_color=COLOR_WARN)
    with q3:
        kpi_card("Doublons d'ID", f"{duplicate_ids:,}", delta=ID_COL, border_color=COLOR_WARN)
    with q4:
        kpi_card("Valeurs manquantes", f"{int(raw_df.isna().sum().sum()):,}", border_color=COLOR_YES)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Types de données")
        dtype_df = pd.DataFrame({
            "Colonne": raw_df.columns,
            "Type": [str(dtype) for dtype in raw_df.dtypes],
        })
        st.dataframe(dtype_df, use_container_width=True, hide_index=True)
    with c2:
        st.markdown("##### Valeurs manquantes par colonne")
        if missing.empty:
            st.success("Aucune valeur manquante détectée dans le fichier brut.")
        else:
            missing_df = missing.rename("Nombre").reset_index(names="Colonne")
            fig_missing = px.bar(missing_df, x="Colonne", y="Nombre", title="Distribution des valeurs manquantes", template=PLOTLY_TEMPLATE)
            st.plotly_chart(fig_missing, use_container_width=True)

    st.markdown("##### Plages observées")
    numeric_cols = raw_df.select_dtypes(include=[np.number]).columns
    range_df = pd.DataFrame({
        "Variable": numeric_cols,
        "Minimum": [raw_df[c].min() for c in numeric_cols],
        "Maximum": [raw_df[c].max() for c in numeric_cols],
        "Médiane": [raw_df[c].median() for c in numeric_cols],
    })
    st.dataframe(range_df, use_container_width=True, hide_index=True)

# ── Tab 1: Numerical distributions ─────────────────────────────────────
if selected_section == "Distributions":
    st.markdown("#### 📈 Distribution des variables numériques par statut d'attrition")
    
    col_sel, col_box = st.columns([2, 3])
    with col_sel:
        selected_num = st.selectbox(
            "Variable numérique à analyser en détail :",
            options=NUMERICAL_FEATURES,
            index=3,  # MonthlyIncome
            key="dist_num_select",
        )
    with col_box:
        st.caption("Survolez les graphiques pour inspecter les valeurs individuelles ou utilisez la loupe pour zoomer.")

    fig_hist = px.histogram(
        filtered,
        x=selected_num,
        color=TARGET,
        barmode="overlay",
        color_discrete_map=COLOR_MAP,
        nbins=24,
        title=f"Distribution & Boîte à moustaches de <b>{selected_num}</b>",
        labels={selected_num: selected_num, "count": "Nombre d'employés", TARGET: "Attrition"},
        opacity=0.75,
        template=PLOTLY_TEMPLATE,
    )
    fig_hist.update_layout(
        bargap=0.04,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60, b=40, l=40, r=40),
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("##### Vue d'ensemble des variables numériques")
    cols_grid = st.columns(3)
    for i, feat in enumerate(NUMERICAL_FEATURES):
        with cols_grid[i % 3]:
            fig_sub = px.histogram(
                filtered,
                x=feat,
                color=TARGET,
                barmode="overlay",
                color_discrete_map=COLOR_MAP,
                nbins=16,
                title=f"<b>{feat}</b>",
                opacity=0.75,
                height=260,
                template=PLOTLY_TEMPLATE,
            )
            fig_sub.update_layout(
                showlegend=False,
                margin=dict(t=40, b=25, l=25, r=25),
                xaxis_title="",
                yaxis_title="",
            )
            st.plotly_chart(fig_sub, use_container_width=True)

# ── Tab 2: Categorical variables ───────────────────────────────────────
if selected_section == "Facteurs catégoriels":
    st.markdown("#### 📊 Répartition des variables catégorielles")
    cat_cols = st.columns(len(CATEGORICAL_FEATURES))

    for idx, cat_feat in enumerate(CATEGORICAL_FEATURES):
        with cat_cols[idx]:
            cross = (
                filtered.groupby([cat_feat, TARGET])
                .size()
                .reset_index(name="Effectif")
            )
            fig_cat = px.bar(
                cross,
                x=cat_feat,
                y="Effectif",
                color=TARGET,
                barmode="group",
                color_discrete_map=COLOR_MAP,
                title=f"Attrition par <b>{cat_feat}</b>",
                labels={"Effectif": "Employés", cat_feat: cat_feat},
                template=PLOTLY_TEMPLATE,
                height=380,
            )
            fig_cat.update_layout(
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(t=50, b=30, l=30, r=30),
            )
            st.plotly_chart(fig_cat, use_container_width=True)

    # Department vs Overtime cross-tab
    st.markdown("##### 🔍 Tri croisé : Département vs Heures Supplémentaires & Attrition")
    cross_dept_ot = (
        filtered.groupby(["Department", "OverTime", TARGET])
        .size()
        .reset_index(name="Effectif")
    )
    fig_cross = px.bar(
        cross_dept_ot,
        x="Department",
        y="Effectif",
        color=TARGET,
        facet_col="OverTime",
        barmode="stack",
        color_discrete_map=COLOR_MAP,
        title="Répartition par Département selon l'OverTime",
        template=PLOTLY_TEMPLATE,
        height=380,
    )
    st.plotly_chart(fig_cross, use_container_width=True)

# ── Tab 3: Correlation Matrix ──────────────────────────────────────────
if selected_section == "Corrélations":
    st.markdown("#### 🔗 Matrice des Corrélations de Pearson")
    numeric_df = filtered[NUMERICAL_FEATURES].copy()
    corr_matrix = numeric_df.corr()

    fig_corr = px.imshow(
        corr_matrix,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Corrélations bivariées entre variables numériques",
        aspect="auto",
        template=PLOTLY_TEMPLATE,
    )
    fig_corr.update_layout(
        height=540,
        margin=dict(t=60, b=40, l=40, r=40),
    )
    st.plotly_chart(fig_corr, use_container_width=True)

# ── Tab 4: Attrition deep-dive ─────────────────────────────────────────
if selected_section == "Taux d'attrition":
    st.markdown("#### 📉 Analyse ciblée du taux de démission")
    col_a1, col_a2 = st.columns(2)

    with col_a1:
        # Attrition rate by department
        dept_attr = (
            filtered.groupby("Department")[TARGET]
            .apply(lambda s: (s == "Yes").mean() * 100)
            .reset_index(name="Taux (%)")
            .sort_values("Taux (%)", ascending=True)
        )
        fig_dept = px.bar(
            dept_attr,
            x="Taux (%)",
            y="Department",
            orientation="h",
            color="Taux (%)",
            color_continuous_scale=["#10B981", "#F59E0B", "#EF4444"],
            title="Taux d'Attrition par Département",
            template=PLOTLY_TEMPLATE,
            height=360,
        )
        fig_dept.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_dept, use_container_width=True)

    with col_a2:
        # Attrition rate by job level
        level_attr = (
            filtered.groupby("JobLevel")[TARGET]
            .apply(lambda s: (s == "Yes").mean() * 100)
            .reset_index(name="Taux (%)")
        )
        fig_level = px.line(
            level_attr,
            x="JobLevel",
            y="Taux (%)",
            markers=True,
            title="Taux d'Attrition selon le Niveau de Poste (JobLevel)",
            template=PLOTLY_TEMPLATE,
            height=360,
        )
        fig_level.update_traces(line=dict(color=PRIMARY_COLOR, width=3), marker=dict(size=9, color=COLOR_YES))
        st.plotly_chart(fig_level, use_container_width=True)

    # WorkLifeBalance vs JobSatisfaction heatmap of attrition rate
    st.markdown("##### 🎯 Taux d'Attrition selon l'Équilibre Vie Pro/Perso et la Satisfaction")
    pivot_wlb_sat = (
        filtered.groupby(["WorkLifeBalance", "JobSatisfaction"])[TARGET]
        .apply(lambda s: (s == "Yes").mean() * 100)
        .unstack()
    )
    fig_heat = px.imshow(
        pivot_wlb_sat,
        labels=dict(x="Satisfaction au Travail", y="Équilibre Vie Pro/Perso", color="Taux (%)"),
        x=[f"Niv {c}" for c in pivot_wlb_sat.columns],
        y=[f"Niv {r}" for r in pivot_wlb_sat.index],
        text_auto=".1f",
        color_continuous_scale="YlOrRd",
        template=PLOTLY_TEMPLATE,
        height=360,
    )
    st.plotly_chart(fig_heat, use_container_width=True)

# ── Tab 5: Detailed Statistics ─────────────────────────────────────────
if selected_section == "Statistiques":
    st.markdown("#### 📋 Statistiques Descriptives & Tests d'Écart")
    
    stat_df = filtered.groupby(TARGET)[NUMERICAL_FEATURES].agg(["mean", "std", "median"])
    st.dataframe(stat_df.T, use_container_width=True)

    st.markdown("##### 📌 Synthèse comparative : Restés (No) vs Partis (Yes)")
    summary_data = []
    for feat in NUMERICAL_FEATURES:
        mean_no = filtered[filtered[TARGET] == "No"][feat].mean() if (filtered[TARGET] == "No").sum() > 0 else 0
        mean_yes = filtered[filtered[TARGET] == "Yes"][feat].mean() if (filtered[TARGET] == "Yes").sum() > 0 else 0
        diff_pct = ((mean_yes - mean_no) / mean_no * 100) if mean_no != 0 else 0
        summary_data.append({
            "Variable": feat,
            "Moyenne (Restés)": round(mean_no, 2),
            "Moyenne (Partis)": round(mean_yes, 2),
            "Écart (%)": round(diff_pct, 1),
        })
    comp_df = pd.DataFrame(summary_data)
    st.dataframe(
        comp_df,
        column_config={
            "Écart (%)": st.column_config.NumberColumn(
                "Écart (%)",
                format="%.1f %%",
            )
        },
        use_container_width=True,
        hide_index=True,
    )

# ── Tests d'hypothèses statistiques ────────────────────────────────────
if selected_section == "Tests statistiques":
    st.markdown("#### 🧪 Validation statistique formelle (α = 0,05)")
    st.caption("Les tests ci-dessous utilisent les données brutes du filtre latéral; les conclusions sont associatives, pas causales.")

    test_tabs = st.tabs(["Shapiro-Wilk", "Chi-deux", "ANOVA / Kruskal-Wallis", "Mann-Whitney"])

    with test_tabs[0]:
        st.markdown("**Test de normalité — H0 : la distribution suit une loi normale**")
        shapiro_rows = []
        for feat in NUMERICAL_FEATURES:
            values = pd.to_numeric(filtered[feat], errors="coerce").dropna()
            sample = values.sample(min(5000, len(values)), random_state=42) if len(values) > 5000 else values
            if len(sample) >= 3:
                stat, p = stats.shapiro(sample)
                shapiro_rows.append({"Variable": feat, "Statistique W": stat, "p-value": p, "Décision": "Rejet H0" if p < 0.05 else "H0 non rejetée"})
        st.dataframe(pd.DataFrame(shapiro_rows).style.format({"Statistique W": "{:.4f}", "p-value": "{:.2e}"}), use_container_width=True, hide_index=True)

    with test_tabs[1]:
        st.markdown("**Test du chi-deux d'indépendance — H0 : aucune association avec Attrition**")
        chi_rows = []
        for feat in CATEGORICAL_FEATURES:
            table = pd.crosstab(filtered[feat], filtered[TARGET])
            if table.shape[0] > 1 and table.shape[1] > 1:
                stat, p, dof, _ = stats.chi2_contingency(table, correction=False)
                n = table.to_numpy().sum()
                r, c = table.shape
                v = np.sqrt(stat / (n * (min(r - 1, c - 1))))
                chi_rows.append({"Variable": feat, "Chi-deux": stat, "ddl": dof, "p-value": p, "V de Cramér": v, "Décision": "Association" if p < 0.05 else "Aucune association"})
        st.dataframe(pd.DataFrame(chi_rows).style.format({"Chi-deux": "{:.3f}", "p-value": "{:.2e}", "V de Cramér": "{:.3f}"}), use_container_width=True, hide_index=True)

    with test_tabs[2]:
        st.markdown("**Comparaison des moyennes/rangs selon Attrition**")
        para_rows, kw_rows = [], []
        groups = [g[NUMERICAL_FEATURES].dropna().to_numpy().ravel() for _, g in filtered.groupby(TARGET)]
        for feat in NUMERICAL_FEATURES:
            samples = [pd.to_numeric(g[feat], errors="coerce").dropna().to_numpy() for _, g in filtered.groupby(TARGET) if len(g[feat].dropna()) > 1]
            if len(samples) == 2:
                try:
                    a_stat, a_p = stats.f_oneway(*samples)
                    para_rows.append({"Variable": feat, "F ANOVA": a_stat, "p-value": a_p, "Décision": "Moyennes différentes" if a_p < 0.05 else "Non significatif"})
                except (ValueError, RuntimeError):
                    pass
                try:
                    k_stat, k_p = stats.kruskal(*samples)
                    kw_rows.append({"Variable": feat, "H Kruskal-Wallis": k_stat, "p-value": k_p, "Décision": "Distributions différentes" if k_p < 0.05 else "Non significatif"})
                except (ValueError, RuntimeError):
                    pass
        st.markdown("##### ANOVA")
        st.dataframe(pd.DataFrame(para_rows).style.format({"F ANOVA": "{:.3f}", "p-value": "{:.2e}"}), use_container_width=True, hide_index=True)
        st.markdown("##### Kruskal-Wallis")
        st.dataframe(pd.DataFrame(kw_rows).style.format({"H Kruskal-Wallis": "{:.3f}", "p-value": "{:.2e}"}), use_container_width=True, hide_index=True)

    with test_tabs[3]:
        st.markdown("**Mann-Whitney U — comparaison des rangs Restés / Partis**")
        mw_rows = []
        no = filtered[filtered[TARGET] == "No"]
        yes = filtered[filtered[TARGET] == "Yes"]
        for feat in NUMERICAL_FEATURES:
            x = pd.to_numeric(no[feat], errors="coerce").dropna()
            y = pd.to_numeric(yes[feat], errors="coerce").dropna()
            if len(x) > 0 and len(y) > 0:
                stat, p = stats.mannwhitneyu(x, y, alternative="two-sided")
                mw_rows.append({"Variable": feat, "U": stat, "p-value": p, "Décision": "Distributions différentes" if p < 0.05 else "Non significatif"})
        st.dataframe(pd.DataFrame(mw_rows).style.format({"U": "{:.1f}", "p-value": "{:.2e}"}), use_container_width=True, hide_index=True)

# ── Tab 6: Filtered Raw Data ───────────────────────────────────────────
if selected_section == "Données filtrées":
    st.markdown(f"#### 🔍 Échantillon des données sélectionnées ({total_filtered:,} lignes)")
    st.dataframe(
        filtered.head(50),
        use_container_width=True,
        hide_index=True,
    )
    
    # Export anonymisé : ni identifiant ni cible ne quittent l'application.
    exported_df = filtered.drop(columns=[
        col for col in ("EmployeeID", "Attrition") if col in filtered.columns
    ])
    csv_bytes = exported_df.to_csv(index=False).encode("utf-8")
    st.caption("L'export exclut volontairement `EmployeeID` et `Attrition`.")
    st.download_button(
        label="📥 Exporter la sélection en CSV (anonymisé)",
        data=csv_bytes,
        file_name="donnees_filtrees_attrition_anonymise.csv",
        mime="text/csv",
    )
