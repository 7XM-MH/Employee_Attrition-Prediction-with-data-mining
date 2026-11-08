"""Page 2 — Individual Prediction: Predict attrition risk for a single employee."""
import streamlit as st
import sys
from pathlib import Path

# Ensure project root is on sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

st.set_page_config(
    page_title="Prédiction Individuelle",
    page_icon="🔮",
    layout="wide",
)

import pandas as pd
import numpy as np
import plotly.graph_objects as go

from app.utils import (
    inject_custom_css,
    render_sidebar_footer,
    load_model,
    metric_card,
    show_missing_artifact_warning,
    get_risk_level,
    get_risk_color,
    generate_recommendations,
    ALL_FEATURES,
    DEPARTMENT_LIST,
)

inject_custom_css()

# ── Sidebar ────────────────────────────────────────────────────────────
st.sidebar.title("🔮 Prédiction")
st.sidebar.info("Remplissez le profil de l'employé et cliquez sur **Prédire**.")
render_sidebar_footer()

# ── Header ─────────────────────────────────────────────────────────────
st.title("🔮 Prédiction Individuelle")
st.markdown(
    "Saisissez les informations d'un employé ci-dessous pour obtenir une prédiction "
    "du risque d'attrition en temps réel."
)
st.markdown("---")

# ── Input form & results columns ──────────────────────────────────────
col_form, col_spacer, col_result = st.columns([5, 1, 6])

with col_form:
    st.subheader("📝 Profil de l'Employé")

    with st.form("prediction_form"):
        fc1, fc2 = st.columns(2)

        with fc1:
            age = st.number_input(
                "Âge", min_value=22, max_value=60, value=35, step=1
            )
            department = st.selectbox("Département", options=DEPARTMENT_LIST)
            job_level = st.selectbox(
                "Niveau de Poste", options=[1, 2, 3, 4, 5], index=1
            )
            years_at_company = st.number_input(
                "Ancienneté (années)", min_value=0, max_value=35, value=5, step=1
            )
            monthly_income = st.number_input(
                "Revenu Mensuel", min_value=3000, max_value=21410, value=8000, step=100
            )
            distance = st.number_input(
                "Distance Domicile (km)", min_value=1.0, max_value=50.0, value=10.0, step=0.5
            )

        with fc2:
            satisfaction_labels = {
                1: "1 — Très insatisfait",
                2: "2 — Insatisfait",
                3: "3 — Neutre",
                4: "4 — Satisfait",
                5: "5 — Très satisfait",
            }
            job_satisfaction = st.select_slider(
                "Satisfaction au Travail",
                options=[1, 2, 3, 4, 5],
                value=3,
                format_func=lambda x: satisfaction_labels[x],
            )

            wlb_labels = {
                1: "1 — Très mauvais",
                2: "2 — Mauvais",
                3: "3 — Bon",
                4: "4 — Excellent",
            }
            work_life_balance = st.select_slider(
                "Équilibre Vie Pro/Perso",
                options=[1, 2, 3, 4],
                value=3,
                format_func=lambda x: wlb_labels[x],
            )

            overtime = st.radio(
                "Heures Supplémentaires",
                options=["Non", "Oui"],
                index=0,
                horizontal=True,
            )

            promotion = st.radio(
                "Promotion (5 dernières années)",
                options=["Non", "Oui"],
                index=0,
                horizontal=True,
            )

            perf_labels = {
                1: "1 — Faible",
                2: "2 — Moyen",
                3: "3 — Bon",
                4: "4 — Excellent",
            }
            performance_rating = st.select_slider(
                "Évaluation de Performance",
                options=[1, 2, 3, 4],
                value=3,
                format_func=lambda x: perf_labels[x],
            )

            training_hours = st.number_input(
                "Heures de Formation (dernière année)",
                min_value=2, max_value=31, value=15, step=1,
            )

        submitted = st.form_submit_button(
            "🔮 Prédire", use_container_width=True, type="primary"
        )

# ── Prediction logic ──────────────────────────────────────────────────
with col_result:
    if submitted:
        model = load_model("best_model")

        if model is None:
            show_missing_artifact_warning(
                "Modèle (best_model.joblib)",
                "python src/train.py",
            )
        else:
            # Map French radio values to English
            overtime_val = "Yes" if overtime == "Oui" else "No"
            promotion_val = "Yes" if promotion == "Oui" else "No"

            # Build input dataframe
            input_data = pd.DataFrame([{
                "Age": age,
                "Department": department,
                "JobLevel": job_level,
                "YearsAtCompany": years_at_company,
                "MonthlyIncome": monthly_income,
                "JobSatisfaction": job_satisfaction,
                "WorkLifeBalance": work_life_balance,
                "OverTime": overtime_val,
                "DistanceFromHome": distance,
                "PromotionLast5Years": promotion_val,
                "PerformanceRating": performance_rating,
                "TrainingHoursLastYear": training_hours,
            }])

            # Predict
            try:
                prediction = model.predict(input_data)[0]
                proba = model.predict_proba(input_data)[0]
                # Find index for "Yes" class
                classes = list(model.classes_)
                yes_idx = classes.index("Yes") if "Yes" in classes else 1
                attrition_proba = float(proba[yes_idx])
            except Exception as e:
                st.error(f"❌ Erreur lors de la prédiction : {e}")
                st.stop()

            risk_label, risk_css, risk_emoji = get_risk_level(attrition_proba)
            risk_color = get_risk_color(attrition_proba)

            # ── Results display ────────────────────────────────────
            st.subheader("📊 Résultat de la Prédiction")

            # Risk badge
            st.markdown(
                f'<div style="text-align:center; margin-bottom:1rem;">'
                f'<span class="risk-badge {risk_css}" style="font-size:1.5rem;">'
                f'{risk_emoji} Risque {risk_label}'
                f'</span></div>',
                unsafe_allow_html=True,
            )

            # Gauge chart
            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=attrition_proba * 100,
                    number={"suffix": "%", "font": {"size": 40}},
                    title={"text": "Probabilité d'Attrition", "font": {"size": 18}},
                    gauge={
                        "axis": {"range": [0, 100], "tickwidth": 2},
                        "bar": {"color": risk_color, "thickness": 0.3},
                        "bgcolor": "white",
                        "steps": [
                            {"range": [0, 30], "color": "#d4edda"},
                            {"range": [30, 60], "color": "#fff3cd"},
                            {"range": [60, 100], "color": "#f8d7da"},
                        ],
                        "threshold": {
                            "line": {"color": risk_color, "width": 4},
                            "thickness": 0.8,
                            "value": attrition_proba * 100,
                        },
                    },
                )
            )
            fig_gauge.update_layout(height=300, margin=dict(t=60, b=20, l=30, r=30))
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Metrics row
            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                metric_card("Prédiction", prediction, "metric-card-red" if prediction == "Yes" else "metric-card-green")
            with mc2:
                metric_card("Probabilité", f"{attrition_proba:.1%}", "metric-card-orange")
            with mc3:
                metric_card("Niveau de Risque", f"{risk_emoji} {risk_label}", f"metric-card-{'red' if risk_label == 'Élevé' else 'orange' if risk_label == 'Modéré' else 'green'}")

            # Recommendations
            st.markdown("---")
            st.subheader("💡 Recommandations de Rétention")
            row_dict = input_data.iloc[0].to_dict()
            recs = generate_recommendations(row_dict, attrition_proba)
            for rec in recs:
                st.markdown(f"- {rec}")

            # SHAP explanation (optional)
            st.markdown("---")
            st.subheader("🔬 Explication de la Prédiction (SHAP)")
            try:
                import shap
                import matplotlib.pyplot as plt

                # Get the preprocessor and classifier from the pipeline
                pipeline = model
                preprocessor = pipeline.named_steps.get("preprocessor") or pipeline[:-1]
                classifier = pipeline.steps[-1][1]

                # Transform input data
                X_transformed = preprocessor.transform(input_data)

                # Get feature names after transformation
                try:
                    feature_names = preprocessor.get_feature_names_out()
                except Exception:
                    feature_names = None

                # Create SHAP explainer
                explainer = shap.Explainer(classifier, feature_names=feature_names)
                shap_values = explainer(X_transformed)

                # For binary classification, use class 1 (Yes) shap values
                if len(shap_values.shape) == 3:
                    shap_values_display = shap_values[:, :, yes_idx]
                else:
                    shap_values_display = shap_values

                fig_shap, ax_shap = plt.subplots(figsize=(10, 6))
                shap.plots.waterfall(shap_values_display[0], max_display=12, show=False)
                plt.title("Contribution des Variables à la Prédiction", fontsize=14)
                plt.tight_layout()
                st.pyplot(fig_shap)
                plt.close(fig_shap)

            except ImportError:
                st.info(
                    "ℹ️ Le module **SHAP** n'est pas installé. "
                    "Installez-le avec `pip install shap` pour voir les explications détaillées."
                )
            except Exception as e:
                st.info(
                    f"ℹ️ L'explication SHAP n'a pas pu être générée : {e}\n\n"
                    "Cela peut arriver avec certains types de modèles."
                )
    else:
        st.subheader("📊 Résultat de la Prédiction")
        st.info(
            "👈 Remplissez le formulaire à gauche et cliquez sur **Prédire** "
            "pour voir les résultats ici."
        )

        # Show example risk levels
        st.markdown("#### Niveaux de Risque")
        st.markdown(
            """
            | Probabilité | Niveau | Indicateur |
            |:-----------:|:------:|:----------:|
            | < 30% | 🟢 Faible | Risque minimal |
            | 30% – 60% | 🟡 Modéré | Surveillance recommandée |
            | ≥ 60% | 🔴 Élevé | Action urgente requise |
            """
        )
