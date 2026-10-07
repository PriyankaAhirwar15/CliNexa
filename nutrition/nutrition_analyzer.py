"""
CliNexa Healthcare Intelligence Platform
Module: Nutrition Analyzer
Description: Interactive food and meal macronutrient analyzer with
comparative target charting, dietary quality indices, and nutritional analytics.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import plotly.graph_objects as go
import plotly.express as px

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "sample" / "nutrition_db.json"


def load_nutrition_database() -> List[Dict[str, Any]]:
    """Load reference food nutrient database."""
    if DB_PATH.exists():
        with open(DB_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("foods", [])
    return []


def analyze_meal_items(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute cumulative nutrient summary for a list of selected food items and quantities.
    Each item in items: {"food_id": str, "servings": float} or explicit nutrient dict.
    """
    db = {f["id"]: f for f in load_nutrition_database()}

    total_cal = 0.0
    total_pro = 0.0
    total_carb = 0.0
    total_fat = 0.0
    total_fib = 0.0
    analyzed_items = []

    for it in items:
        food_id = it.get("food_id")
        servings = float(it.get("servings", 1.0))
        food = db.get(food_id)
        if food:
            cal = food["calories"] * servings
            pro = food["protein_g"] * servings
            carb = food["carbs_g"] * servings
            fat = food["fat_g"] * servings
            fib = food["fiber_g"] * servings

            total_cal += cal
            total_pro += pro
            total_carb += carb
            total_fat += fat
            total_fib += fib

            analyzed_items.append({
                "name": food["name"],
                "servings": servings,
                "serving_size": food.get("serving_size", ""),
                "calories": round(cal, 1),
                "protein_g": round(pro, 1),
                "carbs_g": round(carb, 1),
                "fat_g": round(fat, 1),
                "fiber_g": round(fib, 1)
            })

    total_macro_cals = (total_pro * 4.0) + (total_carb * 4.0) + (total_fat * 9.0)
    pro_pct = round((total_pro * 4.0 / total_macro_cals * 100), 1) if total_macro_cals > 0 else 0
    carb_pct = round((total_carb * 4.0 / total_macro_cals * 100), 1) if total_macro_cals > 0 else 0
    fat_pct = round((total_fat * 9.0 / total_macro_cals * 100), 1) if total_macro_cals > 0 else 0

    return {
        "items": analyzed_items,
        "totals": {
            "calories": round(total_cal),
            "protein_g": round(total_pro, 1),
            "carbs_g": round(total_carb, 1),
            "fat_g": round(total_fat, 1),
            "fiber_g": round(total_fib, 1)
        },
        "macro_distribution_pct": {
            "protein": pro_pct,
            "carbohydrates": carb_pct,
            "fats": fat_pct
        },
        "disclaimer": "Nutritional values are approximate scientific estimates. Variances exist according to raw cultivar, brand, and cooking method."
    }


def create_macro_pie_chart(protein_g: float, carbs_g: float, fat_g: float) -> go.Figure:
    """Create an interactive donut chart of macronutrient caloric contribution."""
    p_cal = protein_g * 4
    c_cal = carbs_g * 4
    f_cal = fat_g * 9
    total = p_cal + c_cal + f_cal

    if total == 0:
        p_cal, c_cal, f_cal = 1, 1, 1

    labels = ["Protein", "Carbohydrates", "Fats"]
    values = [p_cal, c_cal, f_cal]
    colors = ["#2563EB", "#10B981", "#F59E0B"]

    fig = go.Figure(data=[
        go.Pie(
            labels=labels,
            values=values,
            hole=0.6,
            marker=dict(colors=colors),
            textinfo="label+percent",
            hoverinfo="label+value+percent",
            hovertemplate="<b>%{label}</b><br>Calories: %{value:.0f} kcal<br>Percentage: %{percent}<extra></extra>"
        )
    ])

    fig.update_layout(
        title="<b>Macronutrient Caloric Breakdown</b>",
        font=dict(family="Arial, sans-serif", size=13),
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
        margin=dict(l=20, r=20, t=40, b=20),
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def create_target_comparison_bar(actual: Dict[str, float], target: Dict[str, float]) -> go.Figure:
    """Create a grouped bar chart comparing actual intake vs target goals."""
    nutrients = ["Calories (x10 kcal)", "Protein (g)", "Carbs (g)", "Fat (g)", "Fiber (g)"]
    actual_vals = [
        actual.get("calories", 0) / 10.0,
        actual.get("protein_g", 0),
        actual.get("carbs_g", 0),
        actual.get("fat_g", 0),
        actual.get("fiber_g", 0)
    ]
    target_vals = [
        target.get("calories_kcal", 2000) / 10.0,
        target.get("protein_g", 100),
        target.get("carbs_g", 250),
        target.get("fat_g", 65),
        target.get("fiber_g", 30)
    ]

    fig = go.Figure(data=[
        go.Bar(name="Target Goal", x=nutrients, y=target_vals, marker_color="#94A3B8"),
        go.Bar(name="Current / Meal Intake", x=nutrients, y=actual_vals, marker_color="#0D9488")
    ])

    fig.update_layout(
        barmode="group",
        title="<b>Nutritional Intake vs Daily Target Comparison</b>",
        xaxis_title="Nutrient Parameter",
        yaxis_title="Amount",
        font=dict(family="Arial, sans-serif", size=13),
        margin=dict(l=20, r=20, t=40, b=20),
        height=340,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig
