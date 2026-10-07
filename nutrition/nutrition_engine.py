"""
CliNexa Healthcare Intelligence Platform
Module: Nutrition Engine
Description: Evidence-based nutritional assessment, BMR/TDEE calculation,
macronutrient distribution, and dietary recommendation engine.

IMPORTANT NOTICE:
This module provides general nutritional information and dietary guidance
for educational and lifestyle purposes only. It is not medical nutrition therapy.
"""

from typing import Dict, List, Any, Optional


def calculate_bmi(height_cm: float, weight_kg: float) -> Dict[str, Any]:
    """
    Calculate Body Mass Index (BMI) and determine WHO standard category.
    Formula: BMI = weight (kg) / (height (m) ^ 2)
    """
    if height_cm <= 0 or weight_kg <= 0:
        return {
            "bmi": 0.0,
            "category": "Invalid Measurements",
            "healthy_range": "18.5 - 24.9 kg/m²",
            "risk_indicator": "Unable to calculate"
        }

    height_m = height_cm / 100.0
    bmi_value = round(weight_kg / (height_m ** 2), 1)

    if bmi_value < 18.5:
        category = "Underweight"
        risk_indicator = "Increased potential risk of nutritional deficiency"
    elif 18.5 <= bmi_value <= 24.9:
        category = "Normal Weight"
        risk_indicator = "Standard healthy range"
    elif 25.0 <= bmi_value <= 29.9:
        category = "Overweight"
        risk_indicator = "Moderate potential cardiometabolic risk indicator"
    elif 30.0 <= bmi_value <= 34.9:
        category = "Obesity Class I"
        risk_indicator = "Elevated cardiometabolic risk indicator"
    elif 35.0 <= bmi_value <= 39.9:
        category = "Obesity Class II"
        risk_indicator = "High cardiometabolic risk indicator"
    else:
        category = "Obesity Class III"
        risk_indicator = "Very high cardiometabolic risk indicator"

    return {
        "bmi": bmi_value,
        "category": category,
        "healthy_range": "18.5 - 24.9 kg/m²",
        "risk_indicator": risk_indicator
    }


def calculate_bmr_and_tdee(
    age: int,
    gender: str,
    height_cm: float,
    weight_kg: float,
    activity_level: str
) -> Dict[str, Any]:
    """
    Calculate Basal Metabolic Rate (BMR) via Mifflin-St Jeor formula
    and Total Daily Energy Expenditure (TDEE).
    """
    # Mifflin-St Jeor Formula
    if gender.lower() == "male":
        bmr = (10.0 * weight_kg) + (6.25 * height_cm) - (5.0 * age) + 5.0
    else:
        # Female or Default
        bmr = (10.0 * weight_kg) + (6.25 * height_cm) - (5.0 * age) - 161.0

    bmr = max(round(bmr), 800)

    # Activity multiplier
    activity_multipliers = {
        "sedentary": 1.2,
        "lightly active": 1.375,
        "moderately active": 1.55,
        "very active": 1.725,
        "extra active": 1.9
    }

    key = activity_level.lower().strip()
    multiplier = activity_multipliers.get(key, 1.375)
    tdee = round(bmr * multiplier)

    return {
        "bmr": bmr,
        "tdee": tdee,
        "activity_level": activity_level,
        "activity_multiplier": multiplier
    }


def compute_nutrition_targets(
    age: int,
    gender: str,
    height_cm: float,
    weight_kg: float,
    activity_level: str,
    dietary_preference: str = "Vegetarian",
    health_conditions: Optional[List[str]] = None,
    allergies: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Generate evidence-based personalized daily nutritional targets.
    """
    health_conditions = health_conditions or []
    allergies = allergies or []

    energy_data = calculate_bmr_and_tdee(age, gender, height_cm, weight_kg, activity_level)
    tdee = energy_data["tdee"]
    bmi_data = calculate_bmi(height_cm, weight_kg)

    # Base Macronutrient Distributions
    # Protein: 1.0 - 1.6 g per kg (20-25% calories)
    # Fats: 25-30% calories (9 kcal/g)
    # Carbs: Remaining 45-55% calories (4 kcal/g)
    protein_ratio = 0.20
    fat_ratio = 0.28
    carbs_ratio = 0.52

    # Adjust for specific conditions
    condition_notes = []
    conditions_lower = [c.lower() for c in health_conditions]

    if any("diabetes" in c or "glycemic" in c for c in conditions_lower):
        carbs_ratio = 0.45
        protein_ratio = 0.25
        fat_ratio = 0.30
        condition_notes.append("Focus on low-glycemic complex carbohydrates and high soluble fiber to assist blood glucose stability.")

    if any("hypertension" in c or "blood pressure" in c for c in conditions_lower):
        condition_notes.append("Adhere to DASH-aligned guidelines: prioritize potassium, magnesium, and dietary calcium while moderating sodium to < 2,300 mg daily.")

    if any("cholesterol" in c or "cardiac" in c or "heart" in c for c in conditions_lower):
        fat_ratio = 0.25
        condition_notes.append("Emphasize monounsaturated and polyunsaturated fats (omega-3s); restrict saturated fat to < 7% of total energy intake.")

    # Calculate Grams
    protein_calories = tdee * protein_ratio
    protein_grams = round(protein_calories / 4.0)

    fat_calories = tdee * fat_ratio
    fat_grams = round(fat_calories / 9.0)

    carbs_calories = tdee * carbs_ratio
    carbs_grams = round(carbs_calories / 4.0)

    # Fiber recommendations: ~14g per 1000 kcal
    fiber_grams = max(round((tdee / 1000.0) * 14.0), 25)

    # Hydration recommendation: ~30-35 ml per kg body weight
    water_liters = round((weight_kg * 0.033), 1)

    # Recommended food groups based on dietary preferences
    food_groups_recommended = [
        "Fresh Vegetables (Leafy greens, cruciferous vegetables, peppers)",
        "Whole Fruits (Berries, apples, citrus fruits with high polyphenols)",
        "Whole Grains (Rolled oats, quinoa, brown rice, whole wheat)",
        "Healthy Fats (Extra virgin olive oil, avocado, chia seeds, flaxseeds)"
    ]

    if dietary_preference.lower() == "vegan":
        food_groups_recommended.append("Plant-Based Proteins (Lentils, chickpeas, tempeh, organic tofu, edamame)")
    elif dietary_preference.lower() == "vegetarian":
        food_groups_recommended.append("Vegetarian Proteins (Lentils, chickpeas, tofu, Greek yogurt, cottage cheese, eggs)")
    else:
        food_groups_recommended.append("Lean Animal and Plant Proteins (Wild-caught salmon, skinless poultry, legumes, eggs)")

    # Foods to limit
    foods_to_limit = [
        "Ultra-processed foods with high refined carbohydrates and added sugars",
        "Trans fats, hydrogenated oils, and commercially fried items",
        "Sugar-sweetened beverages and concentrated fruit syrups",
        "Excessive dietary sodium from packaged seasonings and processed meats",
        "Heavy alcohol intake and refined confectionery"
    ]

    return {
        "bmi_info": bmi_data,
        "energy_info": energy_data,
        "daily_targets": {
            "calories_kcal": tdee,
            "protein_g": protein_grams,
            "carbs_g": carbs_grams,
            "fat_g": fat_grams,
            "fiber_g": fiber_grams,
            "water_liters": water_liters,
            "sodium_max_mg": 2300
        },
        "macronutrient_split_percent": {
            "protein": round(protein_ratio * 100),
            "carbohydrates": round(carbs_ratio * 100),
            "fats": round(fat_ratio * 100)
        },
        "recommended_food_groups": food_groups_recommended,
        "foods_to_limit": foods_to_limit,
        "condition_notes": condition_notes,
        "allergens_managed": allergies,
        "disclaimer": (
            "This application provides AI-assisted health information and general nutrition "
            "guidance for educational purposes only. It is not a medical diagnosis or a substitute "
            "for professional medical advice. Please consult a qualified healthcare professional "
            "or registered dietitian for individualized clinical nutrition management."
        )
    }
