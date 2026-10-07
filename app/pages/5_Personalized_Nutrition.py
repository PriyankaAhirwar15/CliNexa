"""
CliNexa Healthcare Intelligence Platform
Module 8: Personalized Nutrition Assistant
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
from nutrition.nutrition_engine import compute_nutrition_targets
from nutrition.nutrition_analyzer import create_macro_pie_chart

st.set_page_config(page_title="Personalized Nutrition | CliNexa", page_icon="🥗", layout="wide")

render_medical_disclaimer_banner()
render_page_header(
    title="Module 8: Personalized Nutrition Assistant",
    subtitle="Evidence-based dietary parameters, macronutrient allocations, and personalized food group guidance.",
    badge="Nutritional Science"
)

profile = get_user_profile()

# Compute nutrition targets
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

# Display Key Energy & Macronutrient Metrics
st.subheader("1. Daily Energy & Macronutrient Targets")
c1, c2, c3, c4 = st.columns(4)
with c1:
    render_metric_card(
        title="Target Daily Calories",
        value=f"{targets['daily_targets']['calories_kcal']} kcal",
        subtitle=f"BMR: {targets['energy_info']['bmr']} kcal | TDEE",
        color="#0284C7"
    )
with c2:
    render_metric_card(
        title="Target Protein",
        value=f"{targets['daily_targets']['protein_g']} g",
        subtitle=f"{targets['macronutrient_split_percent']['protein']}% of daily calories",
        color="#10B981"
    )
with c3:
    render_metric_card(
        title="Complex Carbs",
        value=f"{targets['daily_targets']['carbs_g']} g",
        subtitle=f"{targets['macronutrient_split_percent']['carbohydrates']}% of daily calories",
        color="#F59E0B"
    )
with c4:
    render_metric_card(
        title="Healthy Fats",
        value=f"{targets['daily_targets']['fat_g']} g",
        subtitle=f"{targets['macronutrient_split_percent']['fats']}% of daily calories",
        color="#6366F1"
    )

st.markdown("<br>", unsafe_allow_html=True)

# Macronutrient Pie Chart and Fiber / Water Guidance
chart_col, info_col = st.columns([1, 1])

with chart_col:
    pie_fig = create_macro_pie_chart(
        protein_g=targets["daily_targets"]["protein_g"],
        carbs_g=targets["daily_targets"]["carbs_g"],
        fat_g=targets["daily_targets"]["fat_g"]
    )
    st.plotly_chart(pie_fig, use_container_width=True)

with info_col:
    st.markdown("#### Hydration & Micronutrient Priorities")
    st.markdown(
        f"""
        <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 18px;">
            <p>💧 <strong>Daily Fluid Target:</strong> {targets['daily_targets']['water_liters']} Liters (~30-35 ml per kg body weight).</p>
            <p>🌾 <strong>Minimum Soluble & Insoluble Fiber:</strong> {targets['daily_targets']['fiber_g']} grams daily (vital for lipid & glucose management).</p>
            <p>🧂 <strong>Maximum Sodium Guideline:</strong> < {targets['daily_targets']['sodium_max_mg']} mg per day (DASH/AHA standard).</p>
            <p>🌿 <strong>Dietary Alignment:</strong> Tailored for <strong>{profile['dietary_preference']}</strong> eating patterns.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    if targets["condition_notes"]:
        st.markdown("##### Condition-Specific Nutritional Adaptations:")
        for note in targets["condition_notes"]:
            st.info(f"📋 {note}")

st.markdown("---")

# Recommended Food Groups vs Foods to Limit
col_rec, col_lim = st.columns(2)

with col_rec:
    st.subheader("2. Recommended Food Groups to Prioritize")
    st.markdown(
        """
        <div style="background-color: #ECFDF5; border-left: 5px solid #10B981; padding: 16px; border-radius: 6px; margin-bottom: 15px;">
            <p style="margin: 0; font-weight: 600; color: #065F46;">Whole-Food Evidence-Based Recommendations:</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    for fg in targets["recommended_food_groups"]:
        st.markdown(f"✅ **{fg}**")

with col_lim:
    st.subheader("3. Foods & Ingredients to Limit")
    st.markdown(
        """
        <div style="background-color: #FEF2F2; border-left: 5px solid #EF4444; padding: 16px; border-radius: 6px; margin-bottom: 15px;">
            <p style="margin: 0; font-weight: 600; color: #991B1B;">Evidence-Based Reductions for Cardiometabolic Health:</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    for fl in targets["foods_to_limit"]:
        st.markdown(f"⚠️ **{fl}**")

st.markdown("---")

# Strict Healthcare Safety Directives
st.markdown(
    """
    <div style="background-color: #FFFBEB; border: 1px solid #FDE68A; border-radius: 8px; padding: 18px;">
        <h5 style="margin: 0 0 8px 0; color: #B45309;">Clinical Safety Directives:</h5>
        <ul style="margin: 0; padding-left: 20px; color: #92400E; font-size: 13.5px; line-height: 1.6;">
            <li><strong>Prescribed Pharmacotherapy:</strong> Never reduce, stop, or alter dosages of prescribed medications without direct instruction from your treating physician.</li>
            <li><strong>Medical Nutrition Therapy:</strong> Patients diagnosed with chronic kidney disease, severe congestive heart failure, or brittle diabetes require specialized registered dietitian supervision.</li>
            <li><strong>Allergy Warning:</strong> Ensure that all prepared meals strictly respect verified allergies and food intolerances.</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True
)

render_footer()
