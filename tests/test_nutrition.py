"""
Unit Tests for CliNexa Nutrition Engine, Meal Planner, and Food Swap
"""

import pytest
from nutrition.nutrition_engine import calculate_bmi, calculate_bmr_and_tdee, compute_nutrition_targets
from nutrition.meal_planner import generate_7_day_meal_plan
from nutrition.food_swap import search_swaps, get_all_swaps


def test_calculate_bmi():
    # Normal weight
    res_normal = calculate_bmi(175, 70)
    assert res_normal["bmi"] == 22.9
    assert res_normal["category"] == "Normal Weight"

    # Overweight
    res_over = calculate_bmi(170, 80)
    assert res_over["bmi"] == 27.7
    assert res_over["category"] == "Overweight"

    # Obese Class I
    res_obese = calculate_bmi(165, 90)
    assert res_obese["bmi"] == 33.1
    assert res_obese["category"] == "Obesity Class I"

    # Invalid input
    res_invalid = calculate_bmi(-10, 0)
    assert res_invalid["bmi"] == 0.0


def test_calculate_bmr_and_tdee():
    # Male: (10 * 75) + (6.25 * 180) - (5 * 30) + 5 = 750 + 1125 - 150 + 5 = 1730
    energy_male = calculate_bmr_and_tdee(30, "Male", 180, 75, "Moderately active")
    assert energy_male["bmr"] == 1730
    assert energy_male["tdee"] == round(1730 * 1.55)

    # Female: (10 * 60) + (6.25 * 165) - (5 * 25) - 161 = 600 + 1031.25 - 125 - 161 = 1345.25 -> 1345
    energy_female = calculate_bmr_and_tdee(25, "Female", 165, 60, "Sedentary")
    assert energy_female["bmr"] == 1345
    assert energy_female["tdee"] == round(1345 * 1.2)


def test_compute_nutrition_targets():
    targets = compute_nutrition_targets(
        age=40,
        gender="Male",
        height_cm=175,
        weight_kg=78,
        activity_level="Moderately active",
        dietary_preference="Vegetarian"
    )
    assert targets["daily_targets"]["calories_kcal"] > 1500
    assert targets["daily_targets"]["protein_g"] > 50
    assert targets["daily_targets"]["fiber_g"] >= 25
    assert len(targets["recommended_food_groups"]) >= 4
    assert len(targets["foods_to_limit"]) >= 3


def test_7_day_meal_planner():
    plan = generate_7_day_meal_plan(
        dietary_preference="Vegetarian",
        target_calories=2000,
        meals_per_day=5
    )
    assert len(plan["days"]) == 7
    for day in plan["days"]:
        assert len(day["meals"]) == 5
        assert day["totals"]["calories"] > 1000
        assert day["totals"]["protein_g"] > 30


def test_food_swaps():
    all_swaps = get_all_swaps()
    assert len(all_swaps) >= 10

    soda_swaps = search_swaps("soda")
    assert len(soda_swaps) >= 1
    assert "Sparkling Water" in soda_swaps[0]["alternative"]
    assert soda_swaps[0]["calorie_savings"] > 0
