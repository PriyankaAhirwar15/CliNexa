"""
CliNexa Healthcare Intelligence Platform
Module 1: Health Profile & Biometrics
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
from app.utils.session_manager import get_user_profile, update_user_profile
from app.utils.validators import validate_biometrics
from nutrition.nutrition_engine import calculate_bmi, calculate_bmr_and_tdee

st.set_page_config(page_title="Health Profile | CliNexa", page_icon="🩺", layout="wide")

render_medical_disclaimer_banner()
render_page_header(
    title="Module 1: Comprehensive Health Profile",
    subtitle="Configure physiological parameters, lifestyle indicators, and clinical background for tailored intelligence.",
    badge="Biometrics & Profile"
)

profile = get_user_profile()

with st.form("health_profile_form"):
    st.subheader("1. Core Biometrics & Demographics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=int(profile["age"]))
    with col2:
        gender = st.selectbox("Biological Sex", options=["Male", "Female"], index=0 if profile["gender"] == "Male" else 1)
    with col3:
        height_cm = st.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=float(profile["height_cm"]), step=0.5)
    with col4:
        weight_kg = st.number_input("Weight (kg)", min_value=20.0, max_value=300.0, value=float(profile["weight_kg"]), step=0.5)

    st.subheader("2. Lifestyle & Daily Habits")
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        activity_level = st.selectbox(
            "Physical Activity Level",
            options=["Sedentary", "Lightly active", "Moderately active", "Very active", "Extra active"],
            index=["Sedentary", "Lightly active", "Moderately active", "Very active", "Extra active"].index(profile["activity_level"])
        )
    with col6:
        dietary_preference = st.selectbox(
            "Dietary Pattern",
            options=["Vegetarian", "Non-vegetarian", "Vegan"],
            index=["Vegetarian", "Non-vegetarian", "Vegan"].index(profile["dietary_preference"])
        )
    with col7:
        sleep_duration = st.number_input("Sleep Duration (hours/night)", min_value=2.0, max_value=18.0, value=float(profile["sleep_duration"]), step=0.5)
    with col8:
        daily_water = st.number_input("Daily Water Intake (L)", min_value=0.5, max_value=10.0, value=float(profile["daily_water_intake"]), step=0.2)

    st.subheader("3. Clinical Background & Exposures")
    col9, col10 = st.columns(2)
    with col9:
        smoking_status = st.selectbox(
            "Smoking Status",
            options=["Non-smoker", "Former smoker", "Occasional smoker", "Current daily smoker"],
            index=["Non-smoker", "Former smoker", "Occasional smoker", "Current daily smoker"].index(profile["smoking_status"]) if profile["smoking_status"] in ["Non-smoker", "Former smoker", "Occasional smoker", "Current daily smoker"] else 0
        )
        alcohol_consumption = st.selectbox(
            "Alcohol Consumption",
            options=["None", "Occasional / Light", "Moderate", "Frequent"],
            index=["None", "Occasional / Light", "Moderate", "Frequent"].index(profile["alcohol_consumption"]) if profile["alcohol_consumption"] in ["None", "Occasional / Light", "Moderate", "Frequent"] else 1
        )
        known_conditions_str = st.text_area(
            "Known Health Conditions (comma separated)",
            value=", ".join(profile["health_conditions"])
        )

    with col10:
        allergies_str = st.text_area(
            "Food Allergies or Sensitivities (comma separated)",
            value=", ".join(profile["food_allergies"]) if profile["food_allergies"] else "None"
        )
        medications_str = st.text_area(
            "Current Medications (comma separated)",
            value=", ".join(profile["current_medications"])
        )

    submitted = st.form_submit_button("Save & Update Health Profile", use_container_width=True)

if submitted:
    is_valid, err_msg = validate_biometrics(age, height_cm, weight_kg, sleep_duration, daily_water)
    if not is_valid:
        st.error(f"Input validation error: {err_msg}")
    else:
        # Parse lists
        conditions_list = [c.strip() for c in known_conditions_str.split(",") if c.strip() and c.strip().lower() != "none"]
        allergies_list = [a.strip() for a in allergies_str.split(",") if a.strip() and a.strip().lower() != "none"]
        meds_list = [m.strip() for m in medications_str.split(",") if m.strip() and m.strip().lower() != "none"]

        update_user_profile({
            "age": age,
            "gender": gender,
            "height_cm": height_cm,
            "weight_kg": weight_kg,
            "activity_level": activity_level,
            "dietary_preference": dietary_preference,
            "food_allergies": allergies_list,
            "health_conditions": conditions_list,
            "current_medications": meds_list,
            "smoking_status": smoking_status,
            "alcohol_consumption": alcohol_consumption,
            "sleep_duration": sleep_duration,
            "daily_water_intake": daily_water
        })
        st.success("Health profile updated successfully across all platform modules!")

# Current Profile Biometric Computation Display
current_p = get_user_profile()
bmi_res = calculate_bmi(current_p["height_cm"], current_p["weight_kg"])
energy_res = calculate_bmr_and_tdee(
    current_p["age"],
    current_p["gender"],
    current_p["height_cm"],
    current_p["weight_kg"],
    current_p["activity_level"]
)

st.markdown("---")
st.subheader("Calculated Biometric Indicators")

mcol1, mcol2, mcol3, mcol4 = st.columns(4)
with mcol1:
    render_metric_card(
        title="Body Mass Index (BMI)",
        value=f"{bmi_res['bmi']} kg/m²",
        subtitle=f"Category: {bmi_res['category']}",
        color="#0284C7"
    )
with mcol2:
    render_metric_card(
        title="Basal Metabolic Rate",
        value=f"{energy_res['bmr']} kcal",
        subtitle="Mifflin-St Jeor Formula",
        color="#10B981"
    )
with mcol3:
    render_metric_card(
        title="Estimated Daily Energy (TDEE)",
        value=f"{energy_res['tdee']} kcal/day",
        subtitle=f"Multiplier: {energy_res['activity_multiplier']}x ({current_p['activity_level']})",
        color="#F59E0B"
    )
with mcol4:
    render_metric_card(
        title="Hydration Objective",
        value=f"{current_p['daily_water_intake']} Liters",
        subtitle="Recommended Baseline: ~2.5 - 3.0 L",
        color="#06B6D4"
    )

st.markdown(
    f"""
    <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 16px; margin-top: 15px;">
        <h5 style="margin: 0 0 6px 0; color: #1E293B;">Biometric Evaluation Context:</h5>
        <p style="margin: 0; color: #475569; font-size: 13.5px; line-height: 1.5;">
            • <strong>BMI Classification:</strong> {bmi_res['category']} ({bmi_res['healthy_range']}).<br>
            • <strong>Health Indicator Status:</strong> {bmi_res['risk_indicator']}.<br>
            • <em>Notice:</em> Body Mass Index is an anthropometric screening indicator and does not directly measure subcutaneous vs visceral adiposity, skeletal muscle mass, or metabolic health. No definitive medical diagnosis is made based solely on BMI.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

render_footer()
