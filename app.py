"""
CliNexa — AI-Powered Healthcare Intelligence Platform
Hugging Face Spaces Native Gradio Application
"""

import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import gradio as gr

# ZeroGPU (HF Spaces) compatibility — import spaces if available
try:
    import spaces
    ON_ZERO_GPU = True
except ImportError:
    # Not on ZeroGPU — create a no-op decorator
    class _FakeSpaces:
        @staticmethod
        def GPU(fn=None, duration=60):
            if fn is not None:
                return fn
            def decorator(f):
                return f
            return decorator
    spaces = _FakeSpaces()
    ON_ZERO_GPU = False

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.risk.risk_dnn import RiskClassifierService
from models.vision.resnet_classifier import MedicalVisionService
from models.nlp.transformer_nlp import BiomedicalNLPService
from explainability.shap_explainer import ClinicalSHAPExplainer
from explainability.gradcam import GradCAMExplainer
from rag.retriever import ClinicalRAGSystem
from nutrition.nutrition_engine import calculate_bmi, calculate_bmr_and_tdee, compute_nutrition_targets
from nutrition.meal_planner import generate_7_day_meal_plan
from nutrition.food_swap import search_swaps, get_categories
from app.utils.document_parser import extract_text_from_file

# Initialize core services
risk_service = RiskClassifierService()
shap_explainer = ClinicalSHAPExplainer(risk_service)
vision_service = MedicalVisionService()
gradcam_explainer = GradCAMExplainer(vision_service)
nlp_service = BiomedicalNLPService()
rag_system = ClinicalRAGSystem()

MANDATORY_DISCLAIMER = (
    "⚠️ **IMPORTANT HEALTHCARE & CLINICAL NOTICE:**\n\n"
    "*This application provides AI-assisted health information and general nutrition guidance "
    "for educational purposes only. It is not a medical diagnosis or a substitute for professional "
    "medical advice. Please consult a qualified healthcare professional for diagnosis and treatment.*"
)

# -------------------------------------------------------------
# Module 1 & 6: Health Profile, Deep Risk DNN, & SHAP Explainability
# -------------------------------------------------------------
@spaces.GPU(duration=60)
def analyze_health_profile(age, gender, height_cm, weight_kg, sys_bp, dia_bp, activity_level, sleep_hrs, water_liters, smoking_status, alcohol_status):
    bmi_res = calculate_bmi(height_cm, weight_kg)
    energy_res = calculate_bmr_and_tdee(int(age), gender, height_cm, weight_kg, activity_level)

    act_map = {"Sedentary": 1.0, "Lightly active": 2.5, "Moderately active": 4.5, "Very active": 7.0, "Extra active": 10.0}
    act_hrs = act_map.get(activity_level, 3.5)

    smoke_val = 0.0 if smoking_status == "Non-smoker" else (1.5 if smoking_status == "Former smoker" else 2.5)
    alc_val = 0.5 if "Occasional" in alcohol_status else (1.5 if "Moderate" in alcohol_status else (2.5 if "Frequent" in alcohol_status else 0.0))

    features = [
        float(age),
        float(bmi_res["bmi"]),
        float(sys_bp),
        float(dia_bp),
        act_hrs,
        float(sleep_hrs),
        smoke_val,
        alc_val,
        float(water_liters),
        0.5
    ]

    risk_res = risk_service.predict(features)
    shap_res = shap_explainer.explain_instance(features)
    shap_chart = shap_explainer.create_shap_bar_chart(shap_res)

    summary_text = (
        f"### 🩺 Biometric & Energy Evaluation\n"
        f"- **BMI:** {bmi_res['bmi']} kg/m² ({bmi_res['category']})\n"
        f"- **Basal Metabolic Rate (BMR):** {energy_res['bmr']} kcal/day\n"
        f"- **Total Daily Energy Expenditure (TDEE):** {energy_res['tdee']} kcal/day\n"
        f"- **Hydration Recommendation:** ~{round(weight_kg * 0.033, 1)} L/day\n\n"
        f"### 🧠 Deep Neural Network Risk Assessment (PyTorch MLP)\n"
        f"- **Overall Risk Indicator:** **{risk_res['overall_risk_score']}%** ({risk_res['risk_category']})\n"
        f"- **Cardiovascular Sub-Index:** {risk_res['cardiovascular_risk_score']}%\n"
        f"- **Metabolic Sub-Index:** {risk_res['metabolic_risk_score']}%\n"
        f"- **Clinical Guidance:** {risk_res['clinical_guidance']}\n\n"
        f"### 🔍 SHAP Explainability Narrative\n"
        f"{shap_res['summary_narrative']}\n\n"
        f"*Note: Zero usage of Logistic Regression, Random Forest, or XGBoost.*"
    )
    return summary_text, shap_chart


# -------------------------------------------------------------
# Module 2: Symptom Analysis (ClinicalBERT)
# -------------------------------------------------------------
def analyze_symptoms_nlp(symptom_text):
    if not symptom_text.strip():
        return "Please enter a description of symptoms."
    res = nlp_service.analyze_symptoms(symptom_text)
    if not res["success"]:
        return f"Error: {res.get('error', 'Unable to parse symptoms.')}"

    symptom_table = ""
    for s in res["identified_symptoms"]:
        symptom_table += f"- **Reported:** {s['patient_term']} | **Clinical Nomenclature:** {s['clinical_nomenclature']} | **Category:** {s['health_category']}\n"

    output = (
        f"### 💬 Symptom NLP Findings (ClinicalBERT Concept Alignment)\n\n"
        f"- **Extracted Duration:** {res['duration']}\n"
        f"- **Severity Indicator:** {res['severity_indicator']}\n"
        f"- **Relevant Health Categories:** {', '.join(res['relevant_categories'])}\n\n"
        f"#### Identified Concepts:\n{symptom_table if symptom_table else 'General constitutional markers detected.'}\n\n"
        f"#### Potential Risk Indicator:\n"
        f"{res['potential_risk_indicator']}\n\n"
        f"#### Clinical Decision-Support Advisory:\n"
        f"{res['recommended_action']}\n\n"
        f"*{res['disclaimer']}*"
    )
    return output


# -------------------------------------------------------------
# Module 3: Medical Text Analysis
# -------------------------------------------------------------
def analyze_report(report_text, file_obj):
    text_to_analyze = ""
    if file_obj is not None:
        try:
            with open(file_obj.name, "rb") as f:
                content = f.read()
            parsed = extract_text_from_file(content, file_obj.name)
            if parsed["success"]:
                text_to_analyze = parsed["text"]
            else:
                return parsed["error"]
        except Exception as e:
            return f"Error reading file: {str(e)}"
    elif report_text.strip():
        text_to_analyze = report_text

    if not text_to_analyze:
        return "Please paste text or upload a medical report file (PDF, TXT, DOCX)."

    res = nlp_service.analyze_medical_report_text(text_to_analyze)
    if not res["success"]:
        return res["error"]

    tests_md = ""
    for t in res["extracted_tests"]:
        tests_md += f"- **{t['test_name']}:** {t['measured_value']} {t['unit']} (Range: {t['reference_range']}) → **{t['status']}**\n"

    meds_md = ""
    for m in res["medications_detected"]:
        meds_md += f"- **{m['medication']}** ({m['dosage']}): {m['drug_class']} — *{m['primary_indication']}*\n"

    output = (
        f"### 📄 Clinical Document Findings\n\n"
        f"#### 1. Laboratory Markers ({len(res['extracted_tests'])} detected, {res['abnormal_findings_count']} out of range):\n"
        f"{tests_md if tests_md else 'No numerical laboratory markers identified.'}\n\n"
        f"#### 2. Detected Pharmacotherapy:\n"
        f"{meds_md if meds_md else 'No medications identified in document text.'}\n\n"
        f"#### 3. Clinical Observations:\n"
        + "\n".join([f"- {obs}" for obs in res["clinical_observations"]]) + "\n\n"
        f"*{res['disclaimer']}*"
    )
    return output


# -------------------------------------------------------------
# Module 4 & 5: Medical Image Analysis & Grad-CAM (ResNet-50)
# -------------------------------------------------------------
def analyze_image_gradcam(image, alpha):
    if image is None:
        return "Please upload an image.", None, None
    if isinstance(image, np.ndarray):
        pil_img = Image.fromarray(image).convert("RGB")
    else:
        pil_img = image.convert("RGB")

    pred_res = vision_service.predict(pil_img)
    grad_res = gradcam_explainer.generate_visualizations(pil_img, alpha=float(alpha))

    summary = (
        f"### 🫁 ResNet-50 Radiograph Finding\n\n"
        f"- **Predicted Classification:** **{pred_res['predicted_class']}**\n"
        f"- **Confidence:** **{pred_res['confidence_percent']}%**\n"
        f"- **Architecture:** ResNet-50 Deep Residual Convolutional Neural Network\n"
        f"- **Explainability Method:** Grad-CAM on `layer4` bottleneck convolutions\n\n"
        f"⚠️ **{grad_res['explanation_notice']}**\n\n"
        f"*{pred_res['disclaimer']}*"
    )
    return summary, grad_res["heatmap_image"], grad_res["overlay_image"]


# -------------------------------------------------------------
# Module 7: Grounded RAG Assistant (FAISS)
# -------------------------------------------------------------
def ask_rag_assistant(query):
    if not query.strip():
        return "Please enter a clinical or health question."
    res = rag_system.answer_query(query)
    if not res["success"]:
        return res["error"]

    sources_md = "\n\n### 📚 Retrieved Reference Literature Citations:\n"
    for idx, s in enumerate(res["sources"]):
        sources_md += f"- **[{idx+1}] {s['document_title']}** (Relevance match: {int(s['score']*100)}%)\n"

    output = f"{res['answer']}\n{sources_md}\n\n🛡️ *{res['safety_disclaimer']}*"
    return output


# -------------------------------------------------------------
# Module 9: 7-Day Meal Planner
# -------------------------------------------------------------
def get_meal_plan(diet_pref, calories, meals_per_day, cuisine):
    plan = generate_7_day_meal_plan(
        dietary_preference=diet_pref,
        target_calories=int(calories),
        meals_per_day=int(meals_per_day),
        cuisine_preference=cuisine
    )
    avg = plan["average_daily_nutrition"]
    res_md = (
        f"### 🥗 7-Day Personalized Meal Plan\n"
        f"**Dietary Pattern:** {diet_pref} | **Cuisine:** {cuisine}\n"
        f"**Average Daily Nutrients:** ~{avg['calories']} kcal | Protein: {avg['protein_g']}g | Carbs: {avg['carbs_g']}g | Fat: {avg['fat_g']}g | Fiber: {avg['fiber_g']}g\n\n"
    )
    for day in plan["days"][:3]:  # Display first 3 days summary
        res_md += f"#### 📅 {day['day']} (Totals: ~{day['totals']['calories']} kcal, {day['totals']['protein_g']}g protein):\n"
        for m in day["meals"]:
            res_md += f"- **{m['slot']}:** {m['name']} (~{m['calories']} kcal, {m['protein']}g protein)\n"
        res_md += "\n"
    res_md += f"*(Remaining 4 days generated according to identical macro proportions)*\n\n*{plan['disclaimer']}*"
    return res_md


# -------------------------------------------------------------
# Module 10: Smart Food Swap
# -------------------------------------------------------------
def get_food_swaps(category, search_query):
    cat_arg = None if category == "All" else category
    swaps = search_swaps(query=search_query, category=cat_arg)
    if not swaps:
        return "No matching food swaps found."
    out_md = f"### 🔄 Healthier Food Substitutions ({len(swaps)} items):\n\n"
    for s in swaps:
        out_md += (
            f"#### 🥪 {s['original']} ➔ {s['alternative']}\n"
            f"- **Calorie Savings:** ~{s['calorie_savings']} kcal (From ~{s['calories_original']} to ~{s['calories_swap']} kcal)\n"
            f"- **Macronutrient Upgrade:** {s['macros_benefit']}\n"
            f"- **Scientific Rationale:** {s['scientific_rationale']}\n\n---\n"
        )
    return out_md


# -------------------------------------------------------------
# Build Gradio UI
# -------------------------------------------------------------
with gr.Blocks(title="CliNexa — AI Healthcare Intelligence Platform", theme=gr.themes.Soft(primary_hue="sky")) as demo:
    gr.Markdown("# ⚕️ CliNexa — AI-Powered Healthcare Intelligence Platform")
    gr.Markdown(
        "**Clinical Decision-Support | Biomedical Transformers (ClinicalBERT) | ResNet-50 Vision | Grad-CAM & SHAP Explainable AI | FAISS Grounded RAG**"
    )
    gr.Markdown(MANDATORY_DISCLAIMER)

    with gr.Tabs():
        # TAB 1: Health Profile & Deep Risk
        with gr.Tab("🩺 Health Profile & Deep Risk (SHAP)"):
            with gr.Row():
                with gr.Column():
                    age_in = gr.Slider(18, 90, value=38, step=1, label="Age (years)")
                    gender_in = gr.Radio(["Male", "Female"], value="Male", label="Biological Sex")
                    height_in = gr.Number(value=175.0, label="Height (cm)")
                    weight_in = gr.Number(value=76.0, label="Weight (kg)")
                    sys_in = gr.Slider(90, 200, value=124, step=1, label="Systolic BP (mmHg)")
                    dia_in = gr.Slider(55, 120, value=82, step=1, label="Diastolic BP (mmHg)")
                with gr.Column():
                    act_in = gr.Dropdown(["Sedentary", "Lightly active", "Moderately active", "Very active", "Extra active"], value="Moderately active", label="Activity Level")
                    sleep_in = gr.Slider(4.0, 10.0, value=7.0, step=0.5, label="Sleep Duration (hrs/night)")
                    water_in = gr.Slider(0.5, 5.0, value=2.5, step=0.1, label="Daily Water (Liters)")
                    smoke_in = gr.Dropdown(["Non-smoker", "Former smoker", "Occasional smoker", "Current daily smoker"], value="Non-smoker", label="Smoking Status")
                    alc_in = gr.Dropdown(["None", "Occasional / Light", "Moderate", "Frequent"], value="Occasional / Light", label="Alcohol Consumption")
            profile_btn = gr.Button("Calculate Biometrics & Assess Risk (PyTorch MLP + SHAP)", variant="primary")
            profile_out = gr.Markdown()
            shap_plot_out = gr.Plot(label="SHAP Feature Attribution Waterfall Chart")
            profile_btn.click(
                analyze_health_profile,
                inputs=[age_in, gender_in, height_in, weight_in, sys_in, dia_in, act_in, sleep_in, water_in, smoke_in, alc_in],
                outputs=[profile_out, shap_plot_out]
            )

        # TAB 2: Symptom Analysis
        with gr.Tab("💬 Symptom Analysis (ClinicalBERT)"):
            symptom_in = gr.Textbox(
                lines=4,
                placeholder="Enter symptoms in natural English, e.g. I have been experiencing persistent fatigue, increased thirst, and frequent urination for the past 3 weeks.",
                label="Patient-Reported Symptoms"
            )
            with gr.Row():
                ex1 = gr.Button("Example: Metabolic Concerns")
                ex2 = gr.Button("Example: Cardiopulmonary Discomfort")
            ex1.click(lambda: "I have been experiencing persistent fatigue, increased thirst, and frequent urination for the past 3 weeks.", outputs=symptom_in)
            ex2.click(lambda: "Mild chest pressure and slight shortness of breath when walking up hills for several days.", outputs=symptom_in)

            symptom_btn = gr.Button("Analyze Symptoms with Clinical Transformer", variant="primary")
            symptom_out = gr.Markdown()
            symptom_btn.click(analyze_symptoms_nlp, inputs=symptom_in, outputs=symptom_out)

        # TAB 3: Medical Report Analysis
        with gr.Tab("📄 Medical Report Analysis"):
            with gr.Row():
                report_text_in = gr.Textbox(
                    lines=8,
                    placeholder="Paste clinical report text or metabolic panel findings here...",
                    label="Clinical Document Text"
                )
                report_file_in = gr.File(label="Or Upload Medical File (PDF, TXT, DOCX)", file_types=[".pdf", ".txt", ".docx"])
            rep_ex_btn = gr.Button("Load Demo Metabolic Panel")
            rep_ex_btn.click(
                lambda: "COMPREHENSIVE METABOLIC PANEL\nFasting Blood Glucose: 142 mg/dL (HIGH)\nHbA1c: 7.4 % (HIGH)\nSerum Creatinine: 1.05 mg/dL\nTotal Cholesterol: 224 mg/dL (HIGH)\nLDL Cholesterol: 142 mg/dL (HIGH)\nSerum Triglycerides: 190 mg/dL (HIGH)\nCurrent Medications:\nMetformin 500mg\nLisinopril 10mg\nPatient reports generalized fatigue and persistent thirst.",
                outputs=report_text_in
            )
            report_btn = gr.Button("Extract Lab Tests, Meds & Abnormal Values", variant="primary")
            report_out = gr.Markdown()
            report_btn.click(analyze_report, inputs=[report_text_in, report_file_in], outputs=report_out)

        # TAB 4: Medical Vision & Grad-CAM
        with gr.Tab("🫁 Radiograph Vision & Grad-CAM (ResNet-50)"):
            with gr.Row():
                img_in = gr.Image(type="pil", label="Upload Chest Radiograph (PNG / JPG)")
                alpha_in = gr.Slider(0.2, 0.8, value=0.5, step=0.05, label="Grad-CAM Overlay Heatmap Transparency")
            
            sample_img_btn = gr.Button("Load Sample Thoracic Radiograph")
            sample_img_path = PROJECT_ROOT / "data" / "sample" / "sample_images" / "pneumonia_chest_xray.png"
            if sample_img_path.exists():
                sample_img_btn.click(lambda: Image.open(sample_img_path), outputs=img_in)

            vision_btn = gr.Button("Classify Radiograph & Compute Grad-CAM", variant="primary")
            vision_out = gr.Markdown()
            with gr.Row():
                heat_img_out = gr.Image(type="pil", label="Grad-CAM Heatmap (Jet Colormap)")
                overlay_img_out = gr.Image(type="pil", label="Superimposed Heatmap Overlay")
            vision_btn.click(analyze_image_gradcam, inputs=[img_in, alpha_in], outputs=[vision_out, heat_img_out, overlay_img_out])

        # TAB 5: Grounded RAG Assistant
        with gr.Tab("🧠 Grounded RAG Assistant (FAISS)"):
            rag_query_in = gr.Textbox(
                lines=2,
                placeholder="Ask an evidence-based clinical question, e.g. What is the physiological role of dietary fiber?",
                label="Healthcare Query"
            )
            with gr.Row():
                rq1 = gr.Button("Role of Dietary Fiber?")
                rq2 = gr.Button("Hypertension Lifestyle Factors?")
                rq3 = gr.Button("HbA1c & Fasting Glucose Reference Limits?")
            rq1.click(lambda: "What is the general physiological role of dietary fiber?", outputs=rag_query_in)
            rq2.click(lambda: "What are common lifestyle factors associated with high blood pressure?", outputs=rag_query_in)
            rq3.click(lambda: "What are the standard diagnostic reference ranges for HbA1c and fasting blood sugar?", outputs=rag_query_in)

            rag_btn = gr.Button("Retrieve Grounded Evidence from WHO / AHA / NIH / CDC", variant="primary")
            rag_out = gr.Markdown()
            rag_btn.click(ask_rag_assistant, inputs=rag_query_in, outputs=rag_out)

        # TAB 6: Nutrition & Meal Planner
        with gr.Tab("🥗 Nutrition & 7-Day Meal Plan"):
            with gr.Row():
                diet_in = gr.Dropdown(["Vegetarian", "Non-vegetarian", "Vegan"], value="Vegetarian", label="Dietary Preference")
                cal_in = gr.Slider(1200, 3500, value=2000, step=50, label="Target Daily Calories (kcal)")
                meals_in = gr.Dropdown([3, 4, 5], value=5, label="Meals per Day")
                cuisine_in = gr.Dropdown(["Mediterranean", "Continental", "Asian", "Global Healthy"], value="Mediterranean", label="Cuisine Preference")
            plan_btn = gr.Button("Generate Personalized Meal Architecture", variant="primary")
            plan_out = gr.Markdown()
            plan_btn.click(get_meal_plan, inputs=[diet_in, cal_in, meals_in, cuisine_in], outputs=plan_out)

        # TAB 7: Smart Food Swap
        with gr.Tab("🔄 Smart Food Swap"):
            with gr.Row():
                swap_cat_in = gr.Dropdown(["All"] + get_categories(), value="All", label="Category Filter")
                swap_search_in = gr.Textbox(placeholder="Search e.g. soda, chips, white bread...", label="Search Food Item")
            swap_btn = gr.Button("Find Healthier Alternatives & Calorie Savings", variant="primary")
            swap_out = gr.Markdown()
            swap_btn.click(get_food_swaps, inputs=[swap_cat_in, swap_search_in], outputs=swap_out)

    gr.Markdown("---")
    gr.Markdown("Built by **Priyanka Ahirwar** • Released under MIT License • Educational & Clinical Decision Support Demonstration")

if __name__ == "__main__":
    # show_api=False disables gradio-client schema generation which crashes
    # on ZeroGPU due to a bool-in-schema bug in gradio_client/utils.py
    # server_name="0.0.0.0" required so ZeroGPU proxy can reach the container
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_api=False,
        share=False,
    )

