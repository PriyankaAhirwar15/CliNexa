"""
CliNexa Healthcare Intelligence Platform
Module 12: Interactive What-If Health & Lifestyle Analysis
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import plotly.graph_objects as go
from app.components.disclaimer import render_medical_disclaimer_banner, render_footer
from app.components.header import render_page_header
from app.components.cards import render_metric_card
from app.utils.session_manager import get_user_profile
from models.risk.risk_dnn import RiskClassifierService
from nutrition.nutrition_engine import calculate_bmi, calculate_bmr_and_tdee

st.set_page_config(page_title="What-If Health Analysis | CliNexa", page_icon="🔮", layout="wide")

render_medical_disclaimer_banner()
render_page_header(
    title="Module 12: Interactive What-If Health Simulation",
    subtitle="Simulate prospective lifestyle and biometric modifications to observe model-estimated changes in cardiometabolic risk trajectories.",
    badge="Deep Model Simulation"
)

profile = get_user_profile()

if "risk_service" not in st.session_state:
    st.session_state.risk_service = RiskClassifierService()

risk_service = st.session_state.risk_service

# Baseline values
base_weight = float(profile["weight_kg"])
base_sleep = float(profile["sleep_duration"])
base_water = float(profile["daily_water_intake"])
activity_map = {"Sedentary": 1.0, "Lightly active": 2.5, "Moderately active": 4.5, "Very active": 7.0, "Extra active": 10.0}
base_activity_hrs = activity_map.get(profile["activity_level"], 3.5)

# Calculate baseline model prediction
base_features = [
    float(profile["age"]),
    calculate_bmi(profile["height_cm"], base_weight)["bmi"],
    float(profile.get("systolic_bp", 124)),
    float(profile.get("diastolic_bp", 82)),
    base_activity_hrs,
    base_sleep,
    0.0 if profile["smoking_status"] == "Non-smoker" else 1.5,
    0.5 if "Occasional" in profile["alcohol_consumption"] else 0.0,
    base_water,
    0.5
]
baseline_result = risk_service.predict(base_features)

st.subheader("Interactive Lifestyle Modification Controls")
st.markdown("Adjust parameters below to evaluate model-estimated health trajectory outcomes:")

col1, col2, col3, col4 = st.columns(4)

with col1:
    sim_weight = st.slider(
        "Simulated Weight (kg):",
        min_value=max(35.0, base_weight - 25.0),
        max_value=min(180.0, base_weight + 25.0),
        value=base_weight,
        step=0.5
    )
with col2:
    sim_activity_hrs = st.slider(
        "Weekly Physical Activity (hrs/week):",
        min_value=0.0,
        max_value=14.0,
        value=base_activity_hrs,
        step=0.5
    )
with col3:
    sim_sleep = st.slider(
        "Nightly Sleep Duration (hours):",
        min_value=4.0,
        max_value=10.0,
        value=base_sleep,
        step=0.5
    )
with col4:
    sim_water = st.slider(
        "Daily Hydration (Liters):",
        min_value=1.0,
        max_value=5.0,
        value=base_water,
        step=0.2
    )

# Compute simulated prediction
sim_bmi = calculate_bmi(profile["height_cm"], sim_weight)["bmi"]
sim_features = [
    float(profile["age"]),
    sim_bmi,
    float(profile.get("systolic_bp", 124)) - (2.0 if sim_weight < base_weight else 0.0),
    float(profile.get("diastolic_bp", 82)) - (1.5 if sim_weight < base_weight else 0.0),
    sim_activity_hrs,
    sim_sleep,
    base_features[6],
    base_features[7],
    sim_water,
    base_features[9]
]
sim_result = risk_service.predict(sim_features)

# Comparative Results
st.markdown("---")
st.subheader("Comparative Model-Estimated Trajectories")

delta_risk = round(sim_result["overall_risk_score"] - baseline_result["overall_risk_score"], 1)
delta_bmi = round(sim_bmi - base_features[1], 1)

rc1, rc2, rc3, rc4 = st.columns(4)
with rc1:
    render_metric_card(
        title="Baseline Estimated Risk",
        value=f"{baseline_result['overall_risk_score']}%",
        subtitle=baseline_result["risk_category"],
        color="#64748B"
    )
with rc2:
    render_metric_card(
        title="Simulated Estimated Risk",
        value=f"{sim_result['overall_risk_score']}%",
        subtitle=sim_result["risk_category"],
        delta=f"Model-estimated change: {delta_risk:+0.1f}%",
        color="#10B981" if delta_risk <= 0 else "#EF4444"
    )
with rc3:
    render_metric_card(
        title="Simulated BMI",
        value=f"{sim_bmi} kg/m²",
        subtitle=calculate_bmi(profile["height_cm"], sim_weight)["category"],
        delta=f"BMI change: {delta_bmi:+0.1f}",
        color="#0284C7"
    )
with rc4:
    sim_energy = calculate_bmr_and_tdee(profile["age"], profile["gender"], profile["height_cm"], sim_weight, "Moderately active")
    render_metric_card(
        title="Simulated TDEE",
        value=f"{sim_energy['tdee']} kcal/day",
        subtitle=f"BMR: {sim_energy['bmr']} kcal",
        color="#F59E0B"
    )

st.markdown("<br>", unsafe_allow_html=True)

# Comparison Bar Chart
fig = go.Figure(data=[
    go.Bar(
        name="Baseline Scenario",
        x=["Overall Risk Score (%)", "Cardiovascular Sub-Index (%)", "Metabolic Sub-Index (%)"],
        y=[baseline_result["overall_risk_score"], baseline_result["cardiovascular_risk_score"], baseline_result["metabolic_risk_score"]],
        marker_color="#94A3B8"
    ),
    go.Bar(
        name="Simulated Scenario",
        x=["Overall Risk Score (%)", "Cardiovascular Sub-Index (%)", "Metabolic Sub-Index (%)"],
        y=[sim_result["overall_risk_score"], sim_result["cardiovascular_risk_score"], sim_result["metabolic_risk_score"]],
        marker_color="#0D9488"
    )
])
fig.update_layout(
    barmode="group",
    title="<b>Baseline vs Simulated Model-Estimated Risk Indices</b>",
    yaxis_title="Estimated Score (%)",
    font=dict(family="Arial, sans-serif", size=13),
    height=340,
    margin=dict(l=20, r=20, t=40, b=20),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)"
)
st.plotly_chart(fig, use_container_width=True)

# Clinical Guidance Statement
st.markdown(
    f"""
    <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 18px;">
        <h5 style="margin: 0 0 6px 0; color: #1E293B;">Scientific Interpretation of Simulated Parameters:</h5>
        <p style="margin: 0 0 8px 0; color: #475569; font-size: 14px; line-height: 1.5;">
            {sim_result['clinical_guidance']}
        </p>
        <p style="margin: 0; color: #64748B; font-size: 13px; font-style: italic;">
            <strong>Important Communication Note:</strong> Results reflect a <strong>model-estimated change</strong> based on statistical correlations in epidemiological datasets. They do not constitute a clinical guarantee that lifestyle adjustments will eliminate or cure any medical condition.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

render_footer()
