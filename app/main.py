"""
CliNexa Healthcare Intelligence Platform
Main Application Portal & Executive Overview
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from app.components.disclaimer import render_medical_disclaimer_banner, render_footer
from app.components.header import render_page_header
from app.components.cards import render_metric_card
from app.utils.session_manager import get_user_profile, init_session_state
from nutrition.nutrition_engine import calculate_bmi

st.set_page_config(
    page_title="CliNexa | Healthcare Intelligence Platform",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

init_session_state()
profile = get_user_profile()
bmi_info = calculate_bmi(profile["height_cm"], profile["weight_kg"])

# Medical Disclaimer Banner
render_medical_disclaimer_banner()

# Page Header
render_page_header(
    title="CliNexa Healthcare Intelligence Platform",
    subtitle="Next-generation clinical decision-support, biomedical transformers, deep computer vision, and explainable AI.",
    badge="Deep Learning & Clinical NLP"
)

# Hero Introduction
st.markdown(
    """
    <div style="background: linear-gradient(135deg, #F8FAFC 0%, #EFF6FF 100%); border: 1px solid #DBEAFE; border-radius: 12px; padding: 24px; margin-bottom: 25px;">
        <h3 style="margin-top: 0; color: #1E3A8A; font-size: 20px;">Welcome to CliNexa</h3>
        <p style="color: #334155; font-size: 15px; line-height: 1.6; margin-bottom: 12px;">
            CliNexa integrates modern deep learning, biomedical transformers, computer vision, and explainable AI to assist users in understanding complex health markers, reviewing medical documents, exploring personalized evidence-aligned nutrition, and querying trusted medical guidelines.
        </p>
        <div style="display: flex; gap: 15px; flex-wrap: wrap; margin-top: 15px;">
            <span style="background: white; border: 1px solid #CBD5E1; padding: 6px 14px; border-radius: 20px; font-size: 13px; color: #475569; font-weight: 500;">
                🧬 Biomedical NLP (ClinicalBERT)
            </span>
            <span style="background: white; border: 1px solid #CBD5E1; padding: 6px 14px; border-radius: 20px; font-size: 13px; color: #475569; font-weight: 500;">
                🫁 ResNet-50 Chest Radiograph Vision
            </span>
            <span style="background: white; border: 1px solid #CBD5E1; padding: 6px 14px; border-radius: 20px; font-size: 13px; color: #475569; font-weight: 500;">
                🔍 Grad-CAM & SHAP Explainability
            </span>
            <span style="background: white; border: 1px solid #CBD5E1; padding: 6px 14px; border-radius: 20px; font-size: 13px; color: #475569; font-weight: 500;">
                📚 Grounded RAG with FAISS
            </span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Biometric & Status Highlights
st.subheader("Current Active Profile Overview")
col1, col2, col3, col4 = st.columns(4)

with col1:
    render_metric_card(
        title="Body Mass Index (BMI)",
        value=f"{bmi_info['bmi']} kg/m²",
        subtitle=bmi_info["category"],
        delta="Standard WHO Envelope: 18.5 - 24.9"
    )

with col2:
    render_metric_card(
        title="Daily Activity",
        value=profile["activity_level"],
        subtitle=f"Sleep: {profile['sleep_duration']} hrs/night",
        color="#10B981"
    )

with col3:
    render_metric_card(
        title="Dietary Preference",
        value=profile["dietary_preference"],
        subtitle=f"Water: {profile['daily_water_intake']} L/day",
        color="#F59E0B"
    )

with col4:
    render_metric_card(
        title="AI Risk Classifier",
        value="Deep Neural Net",
        subtitle="PyTorch Multi-Head MLP",
        delta="Explainable via SHAP",
        color="#6366F1"
    )

st.markdown("<br>", unsafe_allow_html=True)

# Navigation Cards Grid
st.subheader("Platform Capabilities & Clinical Modules")
grid_col1, grid_col2, grid_col3 = st.columns(3)

with grid_col1:
    st.markdown(
        """
        <div style="border: 1px solid #E2E8F0; border-radius: 10px; padding: 18px; background: white; margin-bottom: 15px;">
            <h4 style="margin: 0 0 8px 0; color: #0284C7;">🩺 1. Health Profile</h4>
            <p style="font-size: 13.5px; color: #475569; line-height: 1.5;">
                Record physiological parameters, biometrics, activity, sleep, and lifestyle habits. Computes BMI and baseline health indicators.
            </p>
        </div>
        <div style="border: 1px solid #E2E8F0; border-radius: 10px; padding: 18px; background: white; margin-bottom: 15px;">
            <h4 style="margin: 0 0 8px 0; color: #0284C7;">💬 2. Symptom Analysis</h4>
            <p style="font-size: 13.5px; color: #475569; line-height: 1.5;">
                Analyze patient symptoms in natural English using biomedical transformers. Extracts medical terms, severity, duration, and health concerns safely.
            </p>
        </div>
        <div style="border: 1px solid #E2E8F0; border-radius: 10px; padding: 18px; background: white; margin-bottom: 15px;">
            <h4 style="margin: 0 0 8px 0; color: #0284C7;">📄 3. Medical Report Analysis</h4>
            <p style="font-size: 13.5px; color: #475569; line-height: 1.5;">
                Upload PDF, DOCX, or TXT laboratory reports. Automatically detects lab tests, abnormal values against reference ranges, and medications.
            </p>
        </div>
        <div style="border: 1px solid #E2E8F0; border-radius: 10px; padding: 18px; background: white;">
            <h4 style="margin: 0 0 8px 0; color: #0284C7;">🫁 4. Medical Image Analysis</h4>
            <p style="font-size: 13.5px; color: #475569; line-height: 1.5;">
                ResNet-50 transfer learning CNN for chest radiographs with Grad-CAM visual heatmaps highlighting model attention regions.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with grid_col2:
    st.markdown(
        """
        <div style="border: 1px solid #E2E8F0; border-radius: 10px; padding: 18px; background: white; margin-bottom: 15px;">
            <h4 style="margin: 0 0 8px 0; color: #0D9488;">🥗 5. Personalized Nutrition</h4>
            <p style="font-size: 13.5px; color: #475569; line-height: 1.5;">
                Science-based BMR and TDEE calculation with condition-aligned macronutrient distribution, recommended food groups, and foods to limit.
            </p>
        </div>
        <div style="border: 1px solid #E2E8F0; border-radius: 10px; padding: 18px; background: white; margin-bottom: 15px;">
            <h4 style="margin: 0 0 8px 0; color: #0D9488;">📅 6. 7-Day Meal Planner</h4>
            <p style="font-size: 13.5px; color: #475569; line-height: 1.5;">
                Structured weekly meal planning for vegetarian, vegan, and non-vegetarian preferences with complete macronutrient breakdowns.
            </p>
        </div>
        <div style="border: 1px solid #E2E8F0; border-radius: 10px; padding: 18px; background: white; margin-bottom: 15px;">
            <h4 style="margin: 0 0 8px 0; color: #0D9488;">🔄 7. Smart Food Swap</h4>
            <p style="font-size: 13.5px; color: #475569; line-height: 1.5;">
                Interactive substitutions of ultra-processed items with nutrient-dense alternatives, including calorie savings and scientific rationale.
            </p>
        </div>
        <div style="border: 1px solid #E2E8F0; border-radius: 10px; padding: 18px; background: white;">
            <h4 style="margin: 0 0 8px 0; color: #0D9488;">📊 8. Nutrition Analyzer</h4>
            <p style="font-size: 13.5px; color: #475569; line-height: 1.5;">
                Interactive meal nutrient calculator with Plotly macronutrient distribution charts and daily target comparison graphs.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with grid_col3:
    st.markdown(
        """
        <div style="border: 1px solid #E2E8F0; border-radius: 10px; padding: 18px; background: white; margin-bottom: 15px;">
            <h4 style="margin: 0 0 8px 0; color: #6366F1;">🔮 9. What-If Health Analysis</h4>
            <p style="font-size: 13.5px; color: #475569; line-height: 1.5;">
                Dynamic simulation sliders to explore model-estimated risk trajectories under prospective changes in weight, sleep, or activity.
            </p>
        </div>
        <div style="border: 1px solid #E2E8F0; border-radius: 10px; padding: 18px; background: white; margin-bottom: 15px;">
            <h4 style="margin: 0 0 8px 0; color: #6366F1;">🧠 10. RAG Health Assistant</h4>
            <p style="font-size: 13.5px; color: #475569; line-height: 1.5;">
                FAISS vector database semantic retrieval over verified clinical references (WHO, AHA, NIH, CDC) for grounded questions.
            </p>
        </div>
        <div style="border: 1px solid #E2E8F0; border-radius: 10px; padding: 18px; background: white; margin-bottom: 15px;">
            <h4 style="margin: 0 0 8px 0; color: #6366F1;">🔍 11. Explainable AI</h4>
            <p style="font-size: 13.5px; color: #475569; line-height: 1.5;">
                SHAP feature attribution charts explaining the positive and negative contributors behind deep learning risk predictions.
            </p>
        </div>
        <div style="border: 1px solid #E2E8F0; border-radius: 10px; padding: 18px; background: white;">
            <h4 style="margin: 0 0 8px 0; color: #6366F1;">📈 12. Executive Dashboard</h4>
            <p style="font-size: 13.5px; color: #475569; line-height: 1.5;">
                Comprehensive healthcare intelligence command center unifying health status, AI findings, and nutrition analytics.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

render_footer()
