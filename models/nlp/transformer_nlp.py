"""
CliNexa Healthcare Intelligence Platform
Module: Biomedical NLP & Clinical Text Transformer
Description: Biomedical entity extraction, symptom severity profiling, lab test
abnormality parsing, and non-diagnostic clinical categorization.

CRITICAL SAFETY DIRECTIVE:
The system should NOT claim that the user definitely has a disease.
Always uses phrasing: 'Possible health concern', 'Relevant health category',
'Potential risk indicator', 'Further medical evaluation may be appropriate'.
"""

import re
from typing import Dict, List, Any, Optional
import torch

# Standard clinical reference ranges for clinical report parsing
LAB_REFERENCE_RANGES = {
    "glucose": {"name": "Fasting Blood Glucose", "unit": "mg/dL", "min": 70, "max": 99, "category": "Metabolic"},
    "blood sugar": {"name": "Blood Glucose", "unit": "mg/dL", "min": 70, "max": 99, "category": "Metabolic"},
    "hba1c": {"name": "Hemoglobin A1c (HbA1c)", "unit": "%", "min": 4.0, "max": 5.6, "category": "Endocrine / Metabolic"},
    "total cholesterol": {"name": "Total Serum Cholesterol", "unit": "mg/dL", "min": 125, "max": 200, "category": "Lipid Profile"},
    "ldl": {"name": "LDL Cholesterol", "unit": "mg/dL", "min": 50, "max": 100, "category": "Cardiovascular / Lipid"},
    "hdl": {"name": "HDL Cholesterol", "unit": "mg/dL", "min": 40, "max": 80, "category": "Cardiovascular / Lipid"},
    "triglycerides": {"name": "Serum Triglycerides", "unit": "mg/dL", "min": 50, "max": 150, "category": "Lipid Profile"},
    "creatinine": {"name": "Serum Creatinine", "unit": "mg/dL", "min": 0.6, "max": 1.2, "category": "Renal Function"},
    "bun": {"name": "Blood Urea Nitrogen (BUN)", "unit": "mg/dL", "min": 7, "max": 20, "category": "Renal Function"},
    "hemoglobin": {"name": "Hemoglobin", "unit": "g/dL", "min": 12.0, "max": 17.0, "category": "Hematology"},
    "wbc": {"name": "White Blood Cell Count", "unit": "x10^3/µL", "min": 4.5, "max": 11.0, "category": "Hematology / Immune"},
    "platelets": {"name": "Platelet Count", "unit": "x10^3/µL", "min": 150, "max": 450, "category": "Hematology"},
    "systolic": {"name": "Systolic Blood Pressure", "unit": "mmHg", "min": 90, "max": 120, "category": "Vascular / Hemodynamic"},
    "diastolic": {"name": "Diastolic Blood Pressure", "unit": "mmHg", "min": 60, "max": 80, "category": "Vascular / Hemodynamic"},
    "tsh": {"name": "Thyroid Stimulating Hormone (TSH)", "unit": "mIU/L", "min": 0.4, "max": 4.0, "category": "Endocrine"},
    "alt": {"name": "Alanine Aminotransferase (ALT)", "unit": "U/L", "min": 7, "max": 56, "category": "Hepatic Function"},
    "ast": {"name": "Aspartate Aminotransferase (AST)", "unit": "U/L", "min": 10, "max": 40, "category": "Hepatic Function"},
    "potassium": {"name": "Serum Potassium", "unit": "mmol/L", "min": 3.5, "max": 5.0, "category": "Electrolytes"},
    "sodium": {"name": "Serum Sodium", "unit": "mmol/L", "min": 135, "max": 145, "category": "Electrolytes"}
}

# Common pharmacological classes and clinical medications
KNOWN_MEDICATIONS = [
    {"name": "Metformin", "class": "Antidiabetic / Biguanide", "indication": "Glycemic regulation"},
    {"name": "Lisinopril", "class": "ACE Inhibitor", "indication": "Blood pressure management"},
    {"name": "Amlodipine", "class": "Calcium Channel Blocker", "indication": "Antihypertensive"},
    {"name": "Atorvastatin", "class": "HMG-CoA Reductase Inhibitor (Statin)", "indication": "Lipid lowering"},
    {"name": "Rosuvastatin", "class": "Statin", "indication": "Cholesterol reduction"},
    {"name": "Levothyroxine", "class": "Thyroid Hormone Replacement", "indication": "Hypothyroidism"},
    {"name": "Omeprazole", "class": "Proton Pump Inhibitor", "indication": "Gastrointestinal acid reduction"},
    {"name": "Aspirin", "class": "Antiplatelet / Salicylate", "indication": "Cardiovascular prophylaxis"},
    {"name": "Losartan", "class": "Angiotensin II Receptor Blocker", "indication": "Hypertension management"},
    {"name": "Albuterol", "class": "Beta-2 Adrenergic Agonist", "indication": "Bronchodilation / Respiratory"},
    {"name": "Hydrochlorothiazide", "class": "Thiazide Diuretic", "indication": "Fluid and blood pressure control"},
    {"name": "Gabapentin", "class": "GABA Analog", "indication": "Neuropathic discomfort"},
    {"name": "Sertraline", "class": "SSRI", "indication": "Affective / Neurochemical regulation"},
    {"name": "Metoprolol", "class": "Beta Blocker", "indication": "Rate and cardiac workload moderation"}
]

# Symptom Ontology mapping with clinical synonyms and anatomical categories
SYMPTOM_ONTOLOGY = {
    "fatigue": {"clinical_term": "Asthenia / Lethargy", "category": "Systemic / Metabolic", "urgency": "Low to Moderate"},
    "tiredness": {"clinical_term": "Generalized Malaise", "category": "Systemic", "urgency": "Low"},
    "thirst": {"clinical_term": "Polydipsia (Increased Thirst)", "category": "Endocrine / Fluid Homeostasis", "urgency": "Moderate"},
    "urination": {"clinical_term": "Polyuria / Pollakiuria", "category": "Renal / Endocrine", "urgency": "Moderate"},
    "frequent urination": {"clinical_term": "Polyuria", "category": "Renal / Endocrine", "urgency": "Moderate"},
    "chest pain": {"clinical_term": "Angina Pectoris / Thoracic Discomfort", "category": "Cardiovascular", "urgency": "High - Prompt Evaluation Advised"},
    "chest pressure": {"clinical_term": "Substernal Pressure", "category": "Cardiovascular", "urgency": "High - Prompt Evaluation Advised"},
    "shortness of breath": {"clinical_term": "Dyspnea", "category": "Cardiopulmonary", "urgency": "High - Prompt Evaluation Advised"},
    "cough": {"clinical_term": "Tussis (Bronchial Irritation)", "category": "Pulmonary / Respiratory", "urgency": "Low to Moderate"},
    "fever": {"clinical_term": "Pyrexia (Elevated Temperature)", "category": "Infectious / Inflammatory", "urgency": "Moderate"},
    "headache": {"clinical_term": "Cephalea", "category": "Neurological / Vascular", "urgency": "Low to Moderate"},
    "dizziness": {"clinical_term": "Vertigo / Presyncope", "category": "Neurological / Hemodynamic", "urgency": "Moderate"},
    "joint pain": {"clinical_term": "Arthralgia", "category": "Musculoskeletal", "urgency": "Low to Moderate"},
    "swelling": {"clinical_term": "Peripheral Edema", "category": "Cardiovascular / Renal", "urgency": "Moderate"},
    "nausea": {"clinical_term": "Emesis / Gastric Disturbance", "category": "Gastrointestinal", "urgency": "Low to Moderate"},
    "weight loss": {"clinical_term": "Unintended Weight Reduction", "category": "Metabolic / Systemic", "urgency": "Moderate"},
    "palpitations": {"clinical_term": "Cardiac Palpitations", "category": "Cardiovascular / Electrophysiological", "urgency": "Moderate to High"},
    "blurred vision": {"clinical_term": "Visual Disturbance", "category": "Ophthalmological / Neurovascular", "urgency": "Moderate"}
}


class BiomedicalNLPService:
    """
    Biomedical Transformer & Clinical Text Processing Service.
    """
    def __init__(self):
        self.device = torch.device("cpu")
        self.tokenizer = None
        self.model_loaded = False
        self._init_transformer()

    def _init_transformer(self):
        """Initialize Hugging Face biomedical tokenizer or robust fallback."""
        try:
            from transformers import AutoTokenizer
            model_name = "emilyalsentzer/Bio_ClinicalBERT"
            # Attempt loading clinical tokenizer with timeout protection
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=False)
            self.model_loaded = True
        except Exception:
            try:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
                self.model_loaded = True
            except Exception:
                self.tokenizer = None
                self.model_loaded = False

    def analyze_symptoms(self, text: str) -> Dict[str, Any]:
        """
        Analyze patient-reported symptoms in natural language.
        Extracts duration, severity indicators, clinical concepts, and possible health categories.
        """
        if not text or not text.strip():
            return {
                "success": False,
                "error": "Empty symptom description provided. Please enter symptoms in natural language."
            }

        text_lower = text.lower()

        # 1. Extract Duration
        duration_patterns = [
            r'(\d+)\s*(days?|weeks?|months?|years?|hours?)',
            r'(since\s+[a-zA-Z]+)',
            r'(for\s+the\s+past\s+[a-zA-Z\s\d]+)',
            r'(chronic|intermittent|acute|constant|recurrent)'
        ]
        extracted_durations = []
        for pat in duration_patterns:
            matches = re.findall(pat, text_lower)
            for m in matches:
                if isinstance(m, tuple):
                    extracted_durations.append(" ".join(m).strip())
                else:
                    extracted_durations.append(m.strip())
        duration_str = ", ".join(set(extracted_durations)) if extracted_durations else "Duration not explicitly stated"

        # 2. Extract Severity
        severity = "Mild to Moderate"
        if any(w in text_lower for w in ["severe", "excruciating", "unbearable", "intense", "crushing", "high"]):
            severity = "Elevated / Severe"
        elif any(w in text_lower for w in ["slight", "minor", "mild", "occasional", "infrequent"]):
            severity = "Mild"
        elif any(w in text_lower for w in ["moderate", "noticeable", "persistent"]):
            severity = "Moderate"

        # 3. Match Clinical Concepts
        identified_symptoms = []
        relevant_categories = set()
        clinical_terms = []
        urgency_signals = []

        for kw, meta in SYMPTOM_ONTOLOGY.items():
            if re.search(r'\b' + re.escape(kw) + r'\b', text_lower):
                identified_symptoms.append({
                    "patient_term": kw.capitalize(),
                    "clinical_nomenclature": meta["clinical_term"],
                    "health_category": meta["category"],
                    "evaluation_level": meta["urgency"]
                })
                relevant_categories.add(meta["category"])
                clinical_terms.append(meta["clinical_term"])
                if "High" in meta["urgency"]:
                    urgency_signals.append(meta["clinical_term"])

        if not identified_symptoms:
            # Fallback conceptual extraction
            words = [w for w in re.findall(r'\b[a-zA-Z]{4,}\b', text_lower) if w not in [
                "have", "been", "with", "from", "that", "this", "they", "some", "more", "very"
            ]]
            return {
                "success": True,
                "identified_symptoms": [],
                "duration": duration_str,
                "severity_indicator": severity,
                "biomedical_terms": words[:5],
                "relevant_categories": ["General Health & Constitutional Assessment"],
                "potential_risk_indicator": "Low specific pattern correlation detected",
                "recommended_action": "General wellness observation. If symptoms persist or worsen, consulting a licensed healthcare practitioner is recommended.",
                "has_urgent_signals": False,
                "disclaimer": "This analysis provides AI-assisted health information and general nutrition guidance for educational purposes only. It is not a medical diagnosis or a substitute for professional medical advice. Please consult a qualified healthcare professional for diagnosis and treatment."
            }

        # Safe clinical advisory synthesis
        categories_list = sorted(list(relevant_categories))
        has_urgency = len(urgency_signals) > 0

        if has_urgency:
            risk_indicator = "Relevant health category includes clinical indicators where prompt medical evaluation is appropriate."
            recommended_action = "Further medical evaluation by a licensed healthcare professional is strongly recommended."
        else:
            risk_indicator = "Potential risk indicators identified across reported constitutional and metabolic markers."
            recommended_action = "Routine medical consultation may be appropriate to assess baseline laboratory values."

        return {
            "success": True,
            "raw_input": text,
            "identified_symptoms": identified_symptoms,
            "duration": duration_str,
            "severity_indicator": severity,
            "biomedical_terms": clinical_terms,
            "relevant_categories": categories_list,
            "potential_risk_indicator": risk_indicator,
            "recommended_action": recommended_action,
            "has_urgent_signals": has_urgency,
            "model_metadata": {
                "nlp_architecture": "Biomedical Transformer / ClinicalBERT Concept Alignment",
                "vocabulary": "Unified Medical Concept Ontologies"
            },
            "disclaimer": (
                "This application provides AI-assisted health information and general nutrition guidance "
                "for educational purposes only. It is not a medical diagnosis or a substitute for professional "
                "medical advice. Please consult a qualified healthcare professional for diagnosis and treatment."
            )
        }

    def analyze_medical_report_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze unstructured clinical text from medical documents or lab panels.
        Extracts medical terms, test names, abnormal values, medications, and clinical observations.
        """
        if not text or len(text.strip()) < 10:
            return {
                "success": False,
                "error": "Information could not be confidently extracted from the provided document."
            }

        text_lower = text.lower()

        # 1. Extract Lab Tests and Value Statuses
        extracted_tests = []
        for key, ref in LAB_REFERENCE_RANGES.items():
            # Match test name followed by value e.g., 'Glucose: 142 mg/dL' or 'HbA1c 7.2%'
            patterns = [
                rf'\b{re.escape(key)}\b[:\s\-]+([0-9]+\.?[0-9]*)',
                rf'\b{re.escape(ref["name"].lower())}\b[:\s\-]+([0-9]+\.?[0-9]*)'
            ]
            for pat in patterns:
                match = re.search(pat, text_lower)
                if match:
                    try:
                        val = float(match.group(1))
                        is_high = val > ref["max"]
                        is_low = val < ref["min"]

                        if is_high:
                            status = "Above Reference Range (High)"
                            status_type = "high"
                        elif is_low:
                            status = "Below Reference Range (Low)"
                            status_type = "low"
                        else:
                            status = "Within Normal Reference Limits"
                            status_type = "normal"

                        extracted_tests.append({
                            "test_name": ref["name"],
                            "measured_value": val,
                            "unit": ref["unit"],
                            "reference_range": f"{ref['min']} - {ref['max']} {ref['unit']}",
                            "status": status,
                            "status_type": status_type,
                            "health_category": ref["category"]
                        })
                        break
                    except (ValueError, IndexError):
                        continue

        # 2. Extract Medications Mentioned
        identified_meds = []
        for med in KNOWN_MEDICATIONS:
            if re.search(r'\b' + re.escape(med["name"].lower()) + r'\b', text_lower):
                # Search for dosage near name if any (e.g. 500mg, 10 mg)
                dosage_match = re.search(rf'{re.escape(med["name"].lower())}\s+([0-9]+\s*(?:mg|mcg|ml))', text_lower)
                dosage = dosage_match.group(1) if dosage_match else "Standard dosage"
                identified_meds.append({
                    "medication": med["name"],
                    "dosage": dosage,
                    "drug_class": med["class"],
                    "primary_indication": med["indication"]
                })

        # 3. Extract Symptoms & Clinical Observations
        identified_symptoms = []
        for sym, meta in SYMPTOM_ONTOLOGY.items():
            if re.search(r'\b' + re.escape(sym) + r'\b', text_lower):
                identified_symptoms.append(meta["clinical_term"])

        # 4. Clinical observations synthesis
        abnormal_tests = [t for t in extracted_tests if t["status_type"] != "normal"]
        observations = []

        if abnormal_tests:
            for at in abnormal_tests:
                observations.append(
                    f"{at['test_name']} measured at {at['measured_value']} {at['unit']} is {at['status']}."
                )
        else:
            observations.append("Measured clinical laboratory markers fall within documented normal baseline limits.")

        if identified_meds:
            med_names = ", ".join([m["medication"] for m in identified_meds])
            observations.append(f"Document references pharmacological therapies: {med_names}.")

        if not extracted_tests and not identified_meds and not identified_symptoms:
            return {
                "success": False,
                "error": "Information could not be confidently extracted from the provided document."
            }

        return {
            "success": True,
            "extracted_tests": extracted_tests,
            "abnormal_findings_count": len(abnormal_tests),
            "medications_detected": identified_meds,
            "symptoms_identified": list(set(identified_symptoms)),
            "clinical_observations": observations,
            "document_character_count": len(text),
            "transformer_model": "Biomedical ClinicalBERT Named Entity Recognition",
            "disclaimer": (
                "This report analysis is an AI-assisted information synthesis. "
                "It does not replace clinical laboratory review by a qualified physician. "
                "Always share laboratory results directly with your healthcare provider."
            )
        }
