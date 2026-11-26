"""Page 2 — Individual Prediction: Predict attrition risk for a single employee."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Ensure project root is on sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

st.set_page_config(
    page_title="Prédiction Individuelle — Attrition RH",
    page_icon="🔮",
    layout="wide",
)

from app.utils import (
    load_css,
    load_bundle,
    load_data,
    kpi_card,
    risk_badge,
    section_header,
    gauge_chart,
    render_sidebar,
    show_missing_artifact_warning,
    generate_recommendations,
    PRIMARY_COLOR,
    SECONDARY_COLOR,
    COLOR_NO,
    COLOR_YES,
    COLOR_WARN,
    ALL_FEATURES,
    FEATURE_COLS,
    DEPARTMENT_LIST,
)

# Inject custom stylesheet
load_css()

# Render shared sidebar
render_sidebar(df=None, show_filters=False)

# ── Header ─────────────────────────────────────────────────────────────
section_header(
    "Prédiction Individuelle & Aide à la Décision RH",
    "Estimez le risque de démission d'un collaborateur et simulez des scénarios de rétention",
    icon="🔮",
)

# Initialize session state for prediction results & what-if
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
if "last_input_data" not in st.session_state:
    st.session_state.last_input_data = None
if "last_explanation" not in st.session_state:
    st.session_state.last_explanation = None

df = load_data()

# Two-column layout: Form (left) | Result & What-if (right)
col_form, col_result = st.columns([5, 6], gap="large")

with col_form:
    st.markdown("#### 📝 Profil du Collaborateur")
    st.caption("Renseignez les paramètres démographiques, professionnels et organisationnels.")

    with st.form("employee_prediction_form", clear_on_submit=False):
        tab_perso, tab_job, tab_env = st.tabs([
            "👤 Poste & Démographie",
            "💼 Rémunération & Expérience",
            "⚖️ Conditions de Travail",
        ])

        with tab_perso:
            f_col1, f_col2 = st.columns(2)
            with f_col1:
                age = st.slider(
                    "Âge de l'employé",
                    min_value=18,
                    max_value=65,
                    value=34,
                    help="Âge chronologique en années",
                )
                department = st.selectbox(
                    "Département",
                    options=DEPARTMENT_LIST,
                    index=1,
                    help="Département ou pôle d'activité principal de rattachement",
                )
            with f_col2:
                job_level = st.select_slider(
                    "Niveau hiérarchique (JobLevel)",
                    options=[1, 2, 3, 4, 5],
                    value=2,
                    format_func=lambda x: f"Niveau {x}" + (" (Junior)" if x == 1 else " (Senior/Lead)" if x >= 4 else " (Intermédiaire)"),
                    help="Position dans l'organigramme (1=Junior, 5=Directeur/Cadre dirigeant)",
                )
                distance = st.slider(
                    "Distance domicile-travail (km)",
                    min_value=1,
                    max_value=60,
                    value=12,
                    help="Distance en kilomètres entre le domicile et le lieu de travail",
                )

        with tab_job:
            j_col1, j_col2 = st.columns(2)
            with j_col1:
                monthly_income = st.number_input(
                    "Revenu mensuel brut (MAD)",
                    min_value=2500,
                    max_value=35000,
                    value=8200,
                    step=500,
                    help="Rémunération mensuelle brute en Dirhams marocains (MAD)",
                )
                years_at_company = st.slider(
                    "Ancienneté dans l'entreprise (années)",
                    min_value=0,
                    max_value=40,
                    value=4,
                    help="Nombre d'années consécutives d'exercice dans la société",
                )
            with j_col2:
                promotion = st.radio(
                    "Promotion durant les 5 dernières années ?",
                    options=["Non", "Oui"],
                    index=0,
                    horizontal=True,
                    help="Indique si le collaborateur a bénéficié d'une promotion interne récente",
                )
                perf_labels = {1: "1 — Insuffisant", 2: "2 — Conforme", 3: "3 — Performant", 4: "4 — Exceptionnel"}
                performance_rating = st.select_slider(
                    "Évaluation de performance annuelle",
                    options=[1, 2, 3, 4],
                    value=3,
                    format_func=lambda x: perf_labels[x],
                    help="Note d'évaluation attribuée lors du dernier entretien annuel",
                )

        with tab_env:
            e_col1, e_col2 = st.columns(2)
            with e_col1:
                overtime = st.radio(
                    "Heures supplémentaires régulières ?",
                    options=["Non", "Oui"],
                    index=1,
                    horizontal=True,
                    help="Le collaborateur effectue-t-il fréquemment des heures supplémentaires ?",
                )
                training_hours = st.slider(
                    "Heures de formation (dernière année)",
                    min_value=0,
                    max_value=40,
                    value=15,
                    help="Nombre d'heures de formation suivies au cours des 12 derniers mois",
                )
            with e_col2:
                sat_labels = {1: "1 — Très faible", 2: "2 — Faible", 3: "3 — Bonne", 4: "4 — Excellente"}
                job_satisfaction = st.select_slider(
                    "Satisfaction au travail",
                    options=[1, 2, 3, 4],
                    value=2,
                    format_func=lambda x: sat_labels[x],
                    help="Auto-évaluation ou score baromètre social de satisfaction professionnelle",
                )
                wlb_labels = {1: "1 — Mauvais", 2: "2 — Passable", 3: "3 — Bon", 4: "4 — Optimal"}
                work_life_balance = st.select_slider(
                    "Équilibre vie professionnelle / personnelle",
                    options=[1, 2, 3, 4],
                    value=2,
                    format_func=lambda x: wlb_labels[x],
                    help="Perception de l'équilibre entre temps de travail et vie privée",
                )

        st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🔮 Lancer l'Évaluation du Risque", type="primary", use_container_width=True)

# ── Prediction Logic ──────────────────────────────────────────────────
with col_result:
    bundle = load_bundle("best_model")
    model = None if bundle is None else bundle["model"]
    threshold = float(bundle["threshold"]) if bundle is not None else None
    yes_idx = 1
    if model is not None:
        classes = list(model.classes_)
        yes_idx = classes.index(1) if 1 in classes else (classes.index("Yes") if "Yes" in classes else 1)

    if submitted:
        st.session_state.pop("last_shap_values", None)
        st.session_state.pop("last_explanation", None)
        if bundle is None:
            show_missing_artifact_warning(
                "Modèle entraîné (best_model.joblib)",
                "python scripts/train_pipeline.py",
            )
        else:
            with st.spinner("⏳ Inférence du modèle Machine Learning en cours..."):
                overtime_val = "Yes" if overtime == "Oui" else "No"
                promotion_val = "Yes" if promotion == "Oui" else "No"

                input_data = pd.DataFrame([{
                    "Age": int(age),
                    "Department": str(department),
                    "JobLevel": int(job_level),
                    "YearsAtCompany": int(years_at_company),
                    "MonthlyIncome": float(monthly_income),
                    "JobSatisfaction": int(job_satisfaction),
                    "WorkLifeBalance": int(work_life_balance),
                    "OverTime": overtime_val,
                    "DistanceFromHome": float(distance),
                    "PromotionLast5Years": promotion_val,
                    "PerformanceRating": int(performance_rating),
                    "TrainingHoursLastYear": int(training_hours),
                }])[FEATURE_COLS]

                try:
                    proba_val = float(model.predict_proba(input_data)[0][yes_idx])
                    decision = int(proba_val >= threshold)
                    st.session_state.last_prediction = {
                        "prediction": decision,
                        "label": "Risque Départ" if decision else "Fidélisé",
                        "proba": proba_val,
                        "threshold": threshold,
                    }
                    st.session_state.last_input_data = input_data
                except Exception as err:
                    st.error(f"❌ Erreur lors du calcul de la prédiction : {err}")

    # Display results if available
    if st.session_state.last_prediction is not None:
        res = st.session_state.last_prediction
        pred_label = res["label"]
        proba_val = res["proba"]
        inp = st.session_state.last_input_data

        st.markdown("#### 📊 Diagnostic Prédictif")
        st.caption(f"Décision prise au seuil optimisé du bundle : **{res['threshold']:.4f}** (probabilité de la classe `Yes`).")
        
        # Risk badge
        st.markdown(
            f"""
            <div style="text-align: center; margin-bottom: 0.75rem;">
                {risk_badge(proba_val, large=True)}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Plotly gauge chart
        fig_g = gauge_chart(proba_val, title="Indice de Risque d'Attrition")
        st.plotly_chart(fig_g, use_container_width=True)

        # KPI Summary cards
        k1, k2, k3 = st.columns(3)
        with k1:
            kpi_card(
                "Décision",
                "Risque Départ" if pred_label == "Yes" else "Fidélisé",
                border_color=COLOR_YES if pred_label == "Yes" else COLOR_NO,
            )
        with k2:
            kpi_card("Probabilité", f"{proba_val * 100:.1f}%", border_color=COLOR_WARN)
        with k3:
            degre = "Critique" if proba_val >= 0.6 else "Vigilance" if proba_val >= 0.3 else "Sain"
            kpi_card("Niveau d'Urgence", degre, border_color=COLOR_YES if proba_val >= 0.6 else COLOR_NO)

        # Retention Recommendations
        st.markdown("##### 💡 Recommandations de Rétention Personnalisées")
        row_dict = inp.iloc[0].to_dict()
        recs = generate_recommendations(row_dict, proba_val)
        for r in recs:
            st.markdown(f"- {r}")

        # ── Bonus: What-If Simulation ─────────────────────────────────
        st.markdown("---")
        st.markdown(
            """
            <div class="whatif-card">
                <div class="whatif-title">
                    <span>⚡</span> <span>Simulation « What-If » (Scénarios de Rétention)</span>
                </div>
                <p style="font-size: 0.85rem; color: #4B5563; margin-bottom: 0.75rem;">
                    Ajustez les leviers RH clés ci-dessous pour observer en direct l'impact sur le risque d'attrition.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        sim_c1, sim_c2 = st.columns(2)
        with sim_c1:
            sim_income_boost = st.select_slider(
                "Augmentation de salaire simulée",
                options=[0, 10, 20, 30, 40],
                value=10,
                format_func=lambda x: f"+{x}%" if x > 0 else "Aucune",
            )
            sim_suppress_ot = st.checkbox(
                "Suppression des heures supplémentaires",
                value=True if row_dict.get("OverTime") == "Yes" else False,
            )
        with sim_c2:
            sim_improve_sat = st.select_slider(
                "Amélioration satisfaction au travail",
                options=[0, 1, 2],
                value=1,
                format_func=lambda x: f"+{x} point(s)" if x > 0 else "Inchangée",
            )
            sim_remote_wlb = st.checkbox("Programme Télétravail (Équilibre +1)", value=True)

        if model is not None:
            sim_data = inp.copy()
            sim_data["MonthlyIncome"] = sim_data["MonthlyIncome"] * (1 + sim_income_boost / 100.0)
            if sim_suppress_ot:
                sim_data["OverTime"] = "No"
            sim_data["JobSatisfaction"] = min(4, int(sim_data["JobSatisfaction"].iloc[0]) + sim_improve_sat)
            if sim_remote_wlb:
                sim_data["WorkLifeBalance"] = min(4, int(sim_data["WorkLifeBalance"].iloc[0]) + 1)

            try:
                sim_proba = float(model.predict_proba(sim_data)[0][yes_idx])
                delta_p = (sim_proba - proba_val) * 100

                sc1, sc2 = st.columns(2)
                with sc1:
                    kpi_card("Risque Actuel", f"{proba_val*100:.1f}%", border_color="#9CA3AF")
                with sc2:
                    delta_str = f"{delta_p:+.1f}%"
                    delta_type = "positive" if delta_p < 0 else "negative"
                    kpi_card(
                        "Risque Après Simulation",
                        f"{sim_proba*100:.1f}%",
                        delta=f"{delta_str} d'impact",
                        delta_type=delta_type,
                        border_color=COLOR_NO if sim_proba < proba_val else COLOR_YES,
                    )
            except Exception:
                pass

        # ── Explicabilité locale (module centralisé src.explainability) ──
        st.markdown("---")
        with st.expander("🔬 Analyse Explicative SHAP (Feature Contributions)", expanded=False):
            if st.button("Générer l'explication SHAP", key="calculate_individual_shap"):
                if bundle is None:
                    show_missing_artifact_warning(
                        "Modèle entraîné (best_model.joblib)",
                        "python scripts/train_pipeline.py",
                    )
                else:
                    with st.spinner("⏳ Calcul des contributions SHAP..."):
                        try:
                            from src.explainability import explain_local

                            if df is not None:
                                background = df.sample(n=min(100, len(df)), random_state=42)[FEATURE_COLS]
                            else:
                                background = inp[FEATURE_COLS]
                            st.session_state.last_explanation = explain_local(
                                bundle, inp.iloc[0], background=background, top_k=3,
                            )
                        except Exception as e:
                            st.info(f"ℹ️ L'explication SHAP détaillée n'est pas disponible pour ce type d'estimateur ({e}).")

            explanation = st.session_state.get("last_explanation")
            if explanation is None:
                st.caption("Le calcul SHAP est lancé uniquement à la demande.")
            else:
                st.caption(
                    f"Moteur : `{explanation['explainer']}` — probabilité `Yes` : "
                    f"{explanation['probability']:.4f} — seuil : {explanation['threshold']:.4f}"
                )
                factors = pd.DataFrame(explanation["top_factors"])
                fig_shap = go.Figure(go.Bar(
                    x=factors["contribution"],
                    y=factors["feature"],
                    orientation="h",
                    marker_color=[COLOR_YES if v > 0 else COLOR_NO for v in factors["contribution"]],
                ))
                fig_shap.update_layout(
                    title="Top 3 facteurs — contribution à la probabilité de départ",
                    template="plotly_white",
                    height=280,
                    margin=dict(t=45, b=30, l=150, r=30),
                    xaxis_title="Contribution SHAP",
                    yaxis_title="",
                )
                st.plotly_chart(fig_shap, use_container_width=True)
                st.dataframe(
                    factors.rename(columns={
                        "feature": "Variable", "contribution": "Contribution", "impact": "Impact",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )

    else:
        st.markdown("#### 📊 Diagnostic Prédictif")
        st.info("👈 Complétez les paramètres du collaborateur dans le formulaire à gauche puis cliquez sur **Lancer l'Évaluation du Risque**.")
        
        st.markdown("##### Échelle d'évaluation du risque")
        st.markdown(
            """
            <div style="background: white; border: 1px solid #E5E7EB; border-radius: 0.75rem; padding: 1rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <span class="risk-badge risk-low">🟢 Faible (&lt; 30%)</span>
                    <span style="font-size: 0.85rem; color: #4B5563;">Situation stable, fidélisation standard</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <span class="risk-badge risk-medium">🟡 Modéré (30% - 60%)</span>
                    <span style="font-size: 0.85rem; color: #4B5563;">Vigilance managériale, entretien d'écoute recommandé</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span class="risk-badge risk-high">🔴 Élevé (&ge; 60%)</span>
                    <span style="font-size: 0.85rem; color: #4B5563;">Risque critique de départ, action corrective prioritaire</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
