"""
CliNexa Healthcare Intelligence Platform
Module 15: Executive Clinical & Healthcare Dashboard
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
from app.components.cards import render_metric_card, render_risk_indicator_box
from app.utils.session_manager import get_user_profile
from nutrition.nutrition_engine import calculate_bmi, compute_nutrition_targets
from nutrition.nutrition_analyzer import create_macro_pie_chart
from models.risk.risk_dnn import RiskClassifierService
from explainability.shap_explainer import ClinicalSHAPExplainer

st.set_page_config(page_title="Executive Dashboard | CliNexa", page_icon="📈", layout="wide")

render_medical_disclaimer_banner()
render_page_header(
    title="Module 15: Executive Healthcare Dashboard",
    subtitle="Unified command center synthesizing health biometrics, AI risk classification, nutrition targets, and clinical insights.",
    badge="Executive Command Center"
)

profile = get_user_profile()
bmi_info = calculate_bmi(profile["height_cm"], profile["weight_kg"])
targets = compute_nutrition_targets(
    age=profile["age"],
    gender=profile["gender"],
    height_cm=profile["height_cm"],
    weight_kg=profile["weight_kg"],
    activity_level=profile["activity_level"],
    dietary_preference=profile["dietary_preference"],
    health_conditions=profile["health_conditions"],
    allergies=profile["food_allergies"]
)

if "risk_service" not in st.session_state:
    st.session_state.risk_service = RiskClassifierService()

risk_service = st.session_state.risk_service
explainer = ClinicalSHAPExplainer(risk_service)

# Compute current risk
act_map = {"Sedentary": 1.0, "Lightly active": 2.5, "Moderately active": 4.5, "Very active": 7.0, "Extra active": 10.0}
act_hrs = act_map.get(profile["activity_level"], 3.5)
features = [
    float(profile["age"]),
    bmi_info["bmi"],
    float(profile.get("systolic_bp", 124)),
    float(profile.get("diastolic_bp", 82)),
    act_hrs,
    float(profile["sleep_duration"]),
    0.0 if profile["smoking_status"] == "Non-smoker" else 1.5,
    0.5 if "Occasional" in profile["alcohol_consumption"] else 0.0,
    float(profile["daily_water_intake"]),
    0.5
]
risk_pred = risk_service.predict(features)
shap_data = explainer.explain_instance(features)

# Section 1: Health Biometrics
st.subheader("1. Health Overview & Vital Biometrics")
h1, h2, h3, h4, h5 = st.columns(5)
with h1:
    render_metric_card("Body Mass Index", f"{bmi_info['bmi']} kg/m²", bmi_info["category"], color="#0284C7")
with h2:
    render_metric_card("Activity Level", profile["activity_level"], f"{act_hrs} hrs/wk physical", color="#10B981")
with h3:
    render_metric_card("Nightly Sleep", f"{profile['sleep_duration']} Hours", "Restorative envelope", color="#6366F1")
with h4:
    render_metric_card("Hydration", f"{profile['daily_water_intake']} Liters", f"Target: ~{targets['daily_targets']['water_liters']} L", color="#06B6D4")
with h5:
    render_metric_card("Daily Target", f"{targets['daily_targets']['calories_kcal']} kcal", f"BMR: {targets['energy_info']['bmr']} kcal", color="#F59E0B")

st.markdown("<br>", unsafe_allow_html=True)

# Section 2: AI Clinical Risk Assessment
st.subheader("2. Deep Neural Network Risk Assessment")
render_risk_indicator_box(
    category=risk_pred["risk_category"],
    score=risk_pred["overall_risk_score"],
    guidance=risk_pred["clinical_guidance"],
    badge_color=risk_pred["badge_color"]
)

# Section 3: Dual Column - Nutrition vs AI Explainability
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("3. Personalized Nutrition Blueprint")
    st.markdown(
        f"""
        <div style="background: white; border: 1px solid #E2E8F0; border-radius: 8px; padding: 16px; margin-bottom: 12px;">
            <p><strong>Dietary Pattern:</strong> {profile['dietary_preference']}</p>
            <p><strong>Target Protein:</strong> {targets['daily_targets']['protein_g']}g | <strong>Carbs:</strong> {targets['daily_targets']['carbs_g']}g | <strong>Fats:</strong> {targets['daily_targets']['fat_g']}g | <strong>Fiber:</strong> {targets['daily_targets']['fiber_g']}g</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    pie_fig = create_macro_pie_chart(
        targets["daily_targets"]["protein_g"],
        targets["daily_targets"]["carbs_g"],
        targets["daily_targets"]["fat_g"]
    )
    st.plotly_chart(pie_fig, use_container_width=True)

with col_right:
    st.subheader("4. SHAP Feature Attribution Insights")
    st.markdown(
        f"""
        <div style="background: white; border: 1px solid #E2E8F0; border-radius: 8px; padding: 16px; margin-bottom: 12px;">
            <p style="font-size: 13.5px; line-height: 1.5; color: #334155;">{shap_data['summary_narrative']}</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    shap_chart = explainer.create_shap_bar_chart(shap_data)
    st.plotly_chart(shap_chart, use_container_width=True)

st.markdown("---")

# Section 5: Recent Platform Activity Summary
st.subheader("5. Integrated Module Status Summary")
s1, s2, s3 = st.columns(3)

with s1:
    st.markdown("##### 💬 Symptom NLP Status")
    latest_sym = st.session_state.get("latest_symptom_analysis")
    if latest_sym and latest_sym.get("success"):
        st.success(f"Latest Analysis: {len(latest_sym['identified_symptoms'])} clinical concepts detected ({latest_sym['severity_indicator']}).")
    else:
        st.info("No active symptom analysis in current session. Run Module 2 to analyze symptoms.")

with s2:
    st.markdown("##### 📄 Clinical Report Status")
    latest_rep = st.session_state.get("latest_report_analysis")
    if latest_rep and latest_rep.get("success"):
        st.success(f"Extracted {len(latest_rep['extracted_tests'])} laboratory tests; {latest_rep['abnormal_findings_count']} out of range.")
    else:
        st.info("No document ingested in current session. Ingest a report in Module 3.")

with s3:
    st.markdown("##### 🫁 ResNet-50 Vision Status")
    latest_vis = st.session_state.get("latest_vision_analysis")
    if latest_vis:
        st.success(f"Classification: {latest_vis['predicted_class']} ({latest_vis['confidence_percent']}%) with Grad-CAM heatmap active.")
    else:
        st.info("No radiograph inspected yet. Test sample chest X-rays in Module 4 & 5.")

render_footer()
