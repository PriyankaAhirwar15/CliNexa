"""
CliNexa Healthcare Intelligence Platform
Module 9: Personalized 7-Day Meal Planner
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
from nutrition.meal_planner import generate_7_day_meal_plan
from nutrition.nutrition_engine import calculate_bmr_and_tdee

st.set_page_config(page_title="7-Day Meal Planner | CliNexa", page_icon="📅", layout="wide")

render_medical_disclaimer_banner()
render_page_header(
    title="Module 9: Personalized 7-Day Meal Planner",
    subtitle="Interactive weekly menu generation tailored to dietary pattern, caloric objective, budget, cuisine, and food preferences.",
    badge="Weekly Meal Architecture"
)

profile = get_user_profile()
energy = calculate_bmr_and_tdee(
    profile["age"], profile["gender"], profile["height_cm"], profile["weight_kg"], profile["activity_level"]
)

# Planner Controls
st.subheader("Customize Meal Plan Parameters")
col1, col2, col3, col4 = st.columns(4)

with col1:
    diet_pref = st.selectbox(
        "Dietary Pattern",
        options=["Vegetarian", "Non-vegetarian", "Vegan"],
        index=["Vegetarian", "Non-vegetarian", "Vegan"].index(profile["dietary_preference"])
    )
with col2:
    cal_target = st.number_input(
        "Target Daily Energy (kcal)",
        min_value=1200,
        max_value=4000,
        value=int(energy["tdee"]),
        step=50
    )
with col3:
    meals_count = st.selectbox("Meals Per Day", options=[3, 4, 5], index=2)
with col4:
    cuisine_pref = st.selectbox("Cuisine Style", options=["Mediterranean", "Continental", "Asian", "Global Healthy"], index=0)

col5, col6 = st.columns(2)
with col5:
    budget_pref = st.selectbox("Budget Preference", options=["Budget-friendly", "Moderate", "Premium"], index=1)
with col6:
    dislikes_input = st.text_input("Foods Disliked or Excluded (optional)", placeholder="e.g. mushrooms, shellfish, cilantro")

gen_plan_btn = st.button("Generate 7-Day Structured Meal Plan", type="primary", use_container_width=True)

# Generate Plan
plan = generate_7_day_meal_plan(
    dietary_preference=diet_pref,
    target_calories=cal_target,
    meals_per_day=meals_count,
    budget_preference=budget_pref,
    cuisine_preference=cuisine_pref,
    allergies=profile["food_allergies"],
    dislikes=[d.strip() for d in dislikes_input.split(",") if d.strip()]
)

# Display Average Summary Cards
st.markdown("---")
st.subheader("Weekly Average Daily Nutrition")
avg_nutr = plan["average_daily_nutrition"]
ac1, ac2, ac3, ac4, ac5 = st.columns(5)
with ac1:
    render_metric_card("Avg Energy", f"{avg_nutr['calories']} kcal", "Daily energy envelope", color="#0284C7")
with ac2:
    render_metric_card("Avg Protein", f"{avg_nutr['protein_g']} g", "Muscle maintenance", color="#10B981")
with ac3:
    render_metric_card("Avg Carbs", f"{avg_nutr['carbs_g']} g", "Complex starches", color="#F59E0B")
with ac4:
    render_metric_card("Avg Fat", f"{avg_nutr['fat_g']} g", "Essential fatty acids", color="#6366F1")
with ac5:
    render_metric_card("Avg Fiber", f"{avg_nutr['fiber_g']} g", "Gastrointestinal health", color="#06B6D4")

st.markdown("<br>", unsafe_allow_html=True)

# 7-Day Tabs
st.subheader("Day-by-Day Detailed Meal Schedule")
day_tabs = st.tabs([d["day"] for d in plan["days"]])

for tab, day_data in zip(day_tabs, plan["days"]):
    with tab:
        d_totals = day_data["totals"]
        st.markdown(
            f"""
            <div style="background: #F1F5F9; border-radius: 8px; padding: 12px 18px; margin-bottom: 16px; display: flex; justify-content: space-between; flex-wrap: wrap;">
                <span><strong>Day Totals:</strong> {d_totals['calories']} kcal</span>
                <span><strong>Protein:</strong> {d_totals['protein_g']}g</span>
                <span><strong>Carbohydrates:</strong> {d_totals['carbs_g']}g</span>
                <span><strong>Fats:</strong> {d_totals['fat_g']}g</span>
                <span><strong>Fiber:</strong> {d_totals['fiber_g']}g</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        for meal in day_data["meals"]:
            st.markdown(
                f"""
                <div style="background: white; border: 1px solid #E2E8F0; border-radius: 8px; padding: 16px; margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <span style="font-size: 15px; font-weight: 700; color: #0284C7;">
                            🍽️ {meal['slot']}: {meal['name']}
                        </span>
                        <span style="background: #E0F2FE; color: #0369A1; padding: 3px 10px; border-radius: 20px; font-size: 12px; font-weight: 600;">
                            ~{meal['calories']} kcal
                        </span>
                    </div>
                    <p style="margin: 0 0 10px 0; color: #475569; font-size: 13.5px; line-height: 1.45;">
                        {meal['desc']}
                    </p>
                    <div style="display: flex; gap: 18px; font-size: 12.5px; color: #64748B;">
                        <span>🥩 <strong>Protein:</strong> {meal['protein']}g</span>
                        <span>🌾 <strong>Carbs:</strong> {meal['carbs']}g</span>
                        <span>🥑 <strong>Fat:</strong> {meal['fat']}g</span>
                        <span>🌱 <strong>Fiber:</strong> {meal['fiber']}g</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

st.markdown(
    f"""
    <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 14px; margin-top: 20px;">
        <p style="margin: 0; color: #64748B; font-size: 13px;">
            ℹ️ <strong>Nutritional Estimation Disclaimer:</strong> {plan['disclaimer']}
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

render_footer()
