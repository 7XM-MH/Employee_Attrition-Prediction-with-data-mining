"""Page 4 — Attrition Factors: Feature importance and SHAP analysis."""
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
    page_title="Facteurs de Démission — Attrition RH",
    page_icon="🔍",
    layout="wide",
)

from app.utils import (
    load_css,
    load_data,
    load_model,
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
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET,
    ALL_FEATURES,
)

# Inject custom stylesheet
load_css()

# Render shared sidebar
render_sidebar(df=None, show_filters=False)
PLOTLY_TEMPLATE = get_plotly_template()

# ── Header ─────────────────────────────────────────────────────────────
section_header(
    "Explicabilité & Facteurs Déterminants de Démission",
    "Comprenez les leviers qui influencent le plus la décision des collaborateurs de quitter l'entreprise",
    icon="🔍",
)

# ── Load artifacts ─────────────────────────────────────────────────────
feature_importances = load_json("feature_importances.json")
df = load_data()


def display_name(key: str) -> str:
    """Get the French display name for a model key."""
    return MODEL_NAMES.get(key, key)


selected_section = st.radio(
    "Section d'analyse",
    ["Importance des variables", "Analyse SHAP", "Leviers d'action RH"],
    horizontal=True,
    label_visibility="collapsed",
)

# ── Tab 1: Feature Importances ─────────────────────────────────────────
if selected_section == "Importance des variables":
    st.markdown("#### 🌲 Importance intrinsèque des caractéristiques (Gini / Impureté)")
    st.caption("Calculée à partir des arbres de décision et forêts aléatoires entraînés sur le jeu de données.")

    if feature_importances is not None and len(feature_importances) > 0:
        available_models = list(feature_importances.keys())
        selected_model = st.selectbox(
            "Algorithme à analyser :",
            options=available_models,
            format_func=display_name,
            key="fi_model_select",
        )

        fi = feature_importances[selected_model]
        fi_df = (
            pd.DataFrame([{"Variable": k, "Importance": v} for k, v in fi.items()])
            .sort_values("Importance", ascending=True)
            .reset_index(drop=True)
        )

        top_var = fi_df.iloc[-1]
        top2_var = fi_df.iloc[-2]

        k1, k2, k3 = st.columns(3)
        with k1:
            kpi_card("1er Facteur Clé", top_var["Variable"], delta=f"{top_var['Importance']*100:.1f}% d'importance", border_color=COLOR_YES)
        with k2:
            kpi_card("2ème Facteur Clé", top2_var["Variable"], delta=f"{top2_var['Importance']*100:.1f}% d'importance", border_color=PRIMARY_COLOR)
        with k3:
            kpi_card("Variables Analysées", f"{len(fi_df)} features", border_color=SECONDARY_COLOR)

        st.markdown("<br>", unsafe_allow_html=True)

        fig_fi = px.bar(
            fi_df,
            x="Importance",
            y="Variable",
            orientation="h",
            title=f"Importance relative des caractéristiques — <b>{display_name(selected_model)}</b>",
            color="Importance",
            color_continuous_scale=["#E0E7FF", "#4F46E5"],
            template=PLOTLY_TEMPLATE,
            height=max(420, len(fi_df) * 32),
        )
        fig_fi.update_layout(
            coloraxis_showscale=False,
            margin=dict(t=50, b=30, l=150, r=30),
            xaxis_title="Score d'importance relative (somme = 1.0)",
            yaxis_title="",
        )
        st.plotly_chart(fig_fi, use_container_width=True)

        with st.expander("📋 Voir le tableau des valeurs d'importance"):
            table_df = fi_df.sort_values("Importance", ascending=False).reset_index(drop=True)
            table_df["Importance (%)"] = table_df["Importance"] * 100
            st.dataframe(
                table_df,
                column_config={
                    "Importance (%)": st.column_config.ProgressColumn(
                        "Poids Relatif",
                        format="%.2f %%",
                        min_value=0,
                        max_value=float(table_df["Importance (%)"].max()),
                    )
                },
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.info(
            "ℹ️ Les données d'importance des variables n'ont pas encore été calculées.\n\n"
            "Exécutez `python scripts/train_pipeline.py` pour générer `results/feature_importances.json`."
        )

# ── Tab 2: SHAP Analysis ───────────────────────────────────────────────
if selected_section == "Analyse SHAP":
    st.markdown("#### 🔬 Explicabilité Globale par Valeurs de Shapley (SHAP)")
    st.caption("Mesure la contribution marginale de chaque modalité à la probabilité de départ.")

    model = load_model("best_model")

    if model is None:
        show_missing_artifact_warning("Modèle (best_model.joblib)", "python scripts/train_pipeline.py")
    elif df is None:
        show_missing_artifact_warning("Données (data/raw/Employee_Attrition_DataSet.csv)")
    else:
        if st.button("Calculer les valeurs SHAP", key="calculate_global_shap"):
            try:
                import shap

                n_sample = min(120, len(df))
                sample_df = df.sample(n=n_sample, random_state=42)
                X_sample = sample_df[ALL_FEATURES]

                pipeline = model
                preprocessor = pipeline.named_steps.get("preprocessor") or pipeline[:-1]
                classifier = pipeline.steps[-1][1]

                X_trans = preprocessor.transform(X_sample)
                try:
                    feat_names = list(preprocessor.get_feature_names_out())
                except Exception:
                    feat_names = None

                classes = list(getattr(classifier, "classes_", [0, 1]))
                yes_idx = classes.index("Yes") if "Yes" in classes else (classes.index(1) if 1 in classes else 1)

                with st.spinner("⏳ Évaluation des valeurs SHAP sur un échantillon représentatif..."):
                    explainer = shap.Explainer(classifier, feature_names=feat_names)
                    shap_values = explainer(X_trans)
                    st.session_state.global_shap_display = (
                        shap_values[:, :, yes_idx] if len(shap_values.shape) == 3 else shap_values
                    )
            except ImportError:
                st.info("ℹ️ La bibliothèque `shap` n'est pas installée (`pip install shap`) pour afficher les graphiques de Shapley.")
            except Exception as e:
                st.info(f"ℹ️ Le calcul des valeurs SHAP n'a pas pu aboutir pour cet algorithme : {e}")

        shap_display = st.session_state.get("global_shap_display")
        if shap_display is None:
            st.caption("Le calcul SHAP est lancé uniquement à la demande.")
        else:
            import matplotlib.pyplot as plt

            c_s1, c_s2 = st.columns(2)
            with c_s1:
                st.markdown("##### Beeswarm Plot (Distribution d'impact)")
                fig_bw, ax_bw = plt.subplots(figsize=(8, 6))
                shap.plots.beeswarm(shap_display, max_display=12, show=False)
                plt.title("Impact des valeurs sur la prédiction de départ", fontsize=12)
                plt.tight_layout()
                st.pyplot(fig_bw)
                plt.close(fig_bw)

            with c_s2:
                st.markdown("##### Bar Plot (|SHAP| Moyen)")
                fig_bar_s, ax_bar_s = plt.subplots(figsize=(8, 6))
                shap.plots.bar(shap_display, max_display=12, show=False)
                plt.title("Importance globale absolue (|SHAP|)", fontsize=12)
                plt.tight_layout()
                st.pyplot(fig_bar_s)
                plt.close(fig_bar_s)

# ── Tab 3: Actionable HR Insights ──────────────────────────────────────
if selected_section == "Leviers d'action RH":
    st.markdown("#### 💡 Synthèse Stratégique & Leviers RH Actionnables")
    
    st.markdown(
        """
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1rem; margin-top: 1rem;">
            <div style="background: white; border: 1px solid #E5E7EB; border-left: 4px solid #EF4444; border-radius: 0.75rem; padding: 1.25rem;">
                <h4 style="margin: 0 0 0.5rem 0; color: #111827;">⏰ 1. Maîtrise des Heures Supplémentaires</h4>
                <p style="font-size: 0.9rem; color: #4B5563; margin: 0; line-height: 1.5;">
                    L'<strong>OverTime</strong> régulier est le déclencheur n°1 d'attrition. Les collaborateurs cumulant des heures supplémentaires présentent un risque de démission multiplié par 2 à 3.
                </p>
                <div style="margin-top: 0.75rem; font-size: 0.8rem; font-weight: 600; color: #EF4444;">
                    Action : Plafonner l'overtime et compenser en temps de récupération.
                </div>
            </div>

            <div style="background: white; border: 1px solid #E5E7EB; border-left: 4px solid #F59E0B; border-radius: 0.75rem; padding: 1.25rem;">
                <h4 style="margin: 0 0 0.5rem 0; color: #111827;">💰 2. Rémunération & Équité Interne</h4>
                <p style="font-size: 0.9rem; color: #4B5563; margin: 0; line-height: 1.5;">
                    Le <strong>MonthlyIncome</strong> présente un effet de seuil marqué. En dessous de la médiane sectorielle, le risque de départ est nettement plus élevé, particulièrement pour les profils techniques.
                </p>
                <div style="margin-top: 0.75rem; font-size: 0.8rem; font-weight: 600; color: #F59E0B;">
                    Action : Réajustement des bas salaires et revalorisation au mérite.
                </div>
            </div>

            <div style="background: white; border: 1px solid #E5E7EB; border-left: 4px solid #4F46E5; border-radius: 0.75rem; padding: 1.25rem;">
                <h4 style="margin: 0 0 0.5rem 0; color: #111827;">⚖️ 3. Équilibre Vie Pro / Vie Privée</h4>
                <p style="font-size: 0.9rem; color: #4B5563; margin: 0; line-height: 1.5;">
                    Un score de <strong>WorkLifeBalance</strong> faible (&le; 2) couplé à une longue distance domicile-travail accélère le désengagement des collaborateurs.
                </p>
                <div style="margin-top: 0.75rem; font-size: 0.8rem; font-weight: 600; color: #4F46E5;">
                    Action : Accords de télétravail flexible (2 à 3 jours par semaine).
                </div>
            </div>

            <div style="background: white; border: 1px solid #E5E7EB; border-left: 4px solid #10B981; border-radius: 0.75rem; padding: 1.25rem;">
                <h4 style="margin: 0 0 0.5rem 0; color: #111827;">🎯 4. Parcours d'Intégration & Fidélisation</h4>
                <p style="font-size: 0.9rem; color: #4B5563; margin: 0; line-height: 1.5;">
                    Les employés ayant moins de 2 ans d'ancienneté (<strong>YearsAtCompany</strong> &le; 2) sont les plus susceptibles de quitter l'entreprise si la satisfaction est basse.
                </p>
                <div style="margin-top: 0.75rem; font-size: 0.8rem; font-weight: 600; color: #10B981;">
                    Action : Programme de parrainage et points d'étape semestriels.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
