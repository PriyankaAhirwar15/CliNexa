"""
CliNexa Healthcare Intelligence Platform
Module 11: Interactive Nutrition Analyzer
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
from nutrition.nutrition_engine import compute_nutrition_targets
from nutrition.nutrition_analyzer import (
    load_nutrition_database,
    analyze_meal_items,
    create_macro_pie_chart,
    create_target_comparison_bar
)

st.set_page_config(page_title="Nutrition Analyzer | CliNexa", page_icon="📊", layout="wide")

render_medical_disclaimer_banner()
render_page_header(
    title="Module 11: Food & Meal Nutrition Analyzer",
    subtitle="Assess macronutrient distribution, caloric density, and comparison against daily personal targets.",
    badge="Nutrient Analytics"
)

profile = get_user_profile()
targets = compute_nutrition_targets(
    age=profile["age"],
    gender=profile["gender"],
    height_cm=profile["height_cm"],
    weight_kg=profile["weight_kg"],
    activity_level=profile["activity_level"]
)

foods_db = load_nutrition_database()
food_options = {f"{f['name']} ({f['category']})": f["id"] for f in foods_db}

st.subheader("Select Foods & Portions to Analyze")
col_sel, col_quick = st.columns([2, 1])

with col_sel:
    selected_labels = st.multiselect(
        "Choose food items from the verified nutritional database:",
        options=list(food_options.keys()),
        default=[
            list(food_options.keys())[0],  # Rolled Oats
            list(food_options.keys())[9],  # Greek yogurt
            list(food_options.keys())[15]  # Blueberries
        ]
    )

with col_quick:
    st.markdown("##### Quick Meal Presets:")
    p1, p2 = st.columns(2)
    preset_items = []
    with p1:
        if st.button("High-Protein Bowl", use_container_width=True):
            selected_labels = [
                k for k in food_options.keys() if "Chicken" in k or "Quinoa" in k or "Broccoli" in k
            ]
    with p2:
        if st.button("Plant-Based Medley", use_container_width=True):
            selected_labels = [
                k for k in food_options.keys() if "Lentils" in k or "Spinach" in k or "Avocado" in k
            ]

# Servings input
meal_entries = []
if selected_labels:
    st.markdown("##### Configure Portions:")
    serving_cols = st.columns(min(len(selected_labels), 4))
    for idx, lbl in enumerate(selected_labels):
        food_id = food_options[lbl]
        col_idx = idx % len(serving_cols)
        with serving_cols[col_idx]:
            srv = st.number_input(f"Portions of {lbl.split('(')[0].strip()}:", min_value=0.5, max_value=5.0, value=1.0, step=0.5, key=f"srv_{food_id}")
            meal_entries.append({"food_id": food_id, "servings": srv})

    # Analyze
    summary = analyze_meal_items(meal_entries)
    totals = summary["totals"]

    st.markdown("---")
    st.subheader("Nutritional Synthesis & Caloric Totals")

    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    with mc1:
        render_metric_card("Total Energy", f"{totals['calories']} kcal", "Cumulative meal calories", color="#0284C7")
    with mc2:
        render_metric_card("Protein", f"{totals['protein_g']} g", f"{summary['macro_distribution_pct']['protein']}% calories", color="#10B981")
    with mc3:
        render_metric_card("Carbohydrates", f"{totals['carbs_g']} g", f"{summary['macro_distribution_pct']['carbohydrates']}% calories", color="#F59E0B")
    with mc4:
        render_metric_card("Fats", f"{totals['fat_g']} g", f"{summary['macro_distribution_pct']['fats']}% calories", color="#6366F1")
    with mc5:
        render_metric_card("Fiber", f"{totals['fiber_g']} g", "Prebiotic dietary fiber", color="#06B6D4")

    st.markdown("<br>", unsafe_allow_html=True)

    # Plotly Charts
    chart1, chart2 = st.columns(2)
    with chart1:
        pie_fig = create_macro_pie_chart(
            protein_g=totals["protein_g"],
            carbs_g=totals["carbs_g"],
            fat_g=totals["fat_g"]
        )
        st.plotly_chart(pie_fig, use_container_width=True)

    with chart2:
        bar_fig = create_target_comparison_bar(
            actual=totals,
            target=targets["daily_targets"]
        )
        st.plotly_chart(bar_fig, use_container_width=True)

    # Table breakdown
    st.subheader("Itemized Portion Breakdown")
    st.table(summary["items"])

    st.info(f"ℹ️ {summary['disclaimer']}")
else:
    st.info("Select one or more items from the nutritional database above to see instant macronutrient analytics.")

render_footer()
