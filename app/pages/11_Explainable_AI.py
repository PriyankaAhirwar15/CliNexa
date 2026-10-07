"""
CliNexa Healthcare Intelligence Platform
Module 6: Explainable AI & SHAP Feature Attributions
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from app.components.disclaimer import render_medical_disclaimer_banner, render_footer
from app.components.header import render_page_header
from app.components.cards import render_metric_card
from app.utils.session_manager import get_user_profile
from models.risk.risk_dnn import RiskClassifierService
from explainability.shap_explainer import ClinicalSHAPExplainer
from nutrition.nutrition_engine import calculate_bmi

st.set_page_config(page_title="Explainable AI | CliNexa", page_icon="🔍", layout="wide")

render_medical_disclaimer_banner()
render_page_header(
    title="Module 6: Explainable AI & SHAP Attribution",
    subtitle="Deconstruct deep neural network risk predictions into quantified feature attributions and directional impacts.",
    badge="SHAP (Shapley Additive Explanations)"
)

profile = get_user_profile()

if "risk_service" not in st.session_state:
    st.session_state.risk_service = RiskClassifierService()

risk_service = st.session_state.risk_service
explainer = ClinicalSHAPExplainer(risk_service)

# Construct user's feature vector
activity_map = {"Sedentary": 1.0, "Lightly active": 2.5, "Moderately active": 4.5, "Very active": 7.0, "Extra active": 10.0}
act_hrs = activity_map.get(profile["activity_level"], 3.5)
bmi_val = calculate_bmi(profile["height_cm"], profile["weight_kg"])["bmi"]

raw_features = [
    float(profile["age"]),
    bmi_val,
    float(profile.get("systolic_bp", 124)),
    float(profile.get("diastolic_bp", 82)),
    act_hrs,
    float(profile["sleep_duration"]),
    0.0 if profile["smoking_status"] == "Non-smoker" else 1.5,
    0.5 if "Occasional" in profile["alcohol_consumption"] else 0.0,
    float(profile["daily_water_intake"]),
    0.5
]

# Run SHAP explanation
with st.spinner("Computing Shapley values and feature contribution gradients..."):
    explanation = explainer.explain_instance(raw_features)

# Metric Summary Cards
st.subheader("Model Prediction & Baseline Reference")
c1, c2, c3 = st.columns(3)
with c1:
    render_metric_card(
        title="Epidemiological Population Baseline",
        value=f"{explanation['baseline_score']}%",
        subtitle="Expected risk for demographic median",
        color="#64748B"
    )
with c2:
    render_metric_card(
        title="Patient Model Estimated Risk",
        value=f"{explanation['predicted_score']}%",
        subtitle="Current multi-factor risk score",
        color="#0284C7"
    )
with c3:
    delta_pts = round(explanation["predicted_score"] - explanation["baseline_score"], 1)
    render_metric_card(
        title="Attributed Net Variance",
        value=f"{delta_pts:+0.1f}%",
        subtitle="Net displacement explained by SHAP",
        color="#EF4444" if delta_pts > 0 else "#10B981"
    )

st.markdown("<br>", unsafe_allow_html=True)

# Human-Readable Clinical Explanation
st.subheader("1. AI Synthesis & Feature Importance Narrative")
st.markdown(
    f"""
    <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 20px; margin-bottom: 20px;">
        <p style="margin: 0 0 10px 0; font-size: 15px; line-height: 1.6; color: #1E293B;">
            {explanation['summary_narrative']}
        </p>
        <hr style="margin: 12px 0; border: none; border-top: 1px solid #E2E8F0;">
        <div style="display: flex; gap: 24px; font-size: 13.5px;">
            <div>
                <span style="color: #DC2626; font-weight: 600;">Elevating Factors (+):</span> {', '.join(explanation['positive_factors']) if explanation['positive_factors'] else 'None detected'}
            </div>
            <div>
                <span style="color: #059669; font-weight: 600;">Protective Factors (-):</span> {', '.join(explanation['protective_factors']) if explanation['protective_factors'] else 'None detected'}
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Plotly Horizontal Attribution Bar Chart
st.subheader("2. Quantified SHAP Contribution Bar Chart")
chart_fig = explainer.create_shap_bar_chart(explanation)
st.plotly_chart(chart_fig, use_container_width=True)

# Itemized Attribution Table
st.subheader("3. Feature Breakdown & Measured Biometrics")
table_data = []
for item in explanation["sorted_attributions"]:
    table_data.append({
        "Biometric / Lifestyle Parameter": item["feature"],
        "Recorded Value": item["current_value"],
        "SHAP Attribution Value": item["shap_value"],
        "Net Percentage Impact": f"{item['impact_percent']:+0.2f}%",
        "Directional Influence": item["direction"]
    })
st.dataframe(table_data, use_container_width=True)

# Disclaimer
st.info(f"ℹ️ {explanation['disclaimer']}")

render_footer()
