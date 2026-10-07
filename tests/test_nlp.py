"""
Unit Tests for Biomedical NLP Service (Symptom & Document Analysis)
"""

import pytest
from models.nlp.transformer_nlp import BiomedicalNLPService


def test_symptom_analysis():
    nlp = BiomedicalNLPService()
    symptom_input = "I have experienced fatigue, persistent thirst, and frequent urination for the past 3 weeks."
    res = nlp.analyze_symptoms(symptom_input)

    assert res["success"] is True
    assert "3 weeks" in res["duration"]
    assert len(res["identified_symptoms"]) >= 2
    assert "Endocrine" in " ".join(res["relevant_categories"]) or "Metabolic" in " ".join(res["relevant_categories"])
    # Non-diagnostic phrasing validation
    assert "You have diabetes" not in res["potential_risk_indicator"]
    assert "Potential risk indicator" in res["potential_risk_indicator"] or "Relevant health category" in res["potential_risk_indicator"]


def test_empty_symptom_handling():
    nlp = BiomedicalNLPService()
    res = nlp.analyze_symptoms("")
    assert res["success"] is False


def test_medical_report_analysis():
    nlp = BiomedicalNLPService()
    report_text = """
    COMPREHENSIVE METABOLIC PANEL
    Fasting Blood Glucose: 145 mg/dL
    HbA1c: 7.2 %
    Serum Creatinine: 1.0 mg/dL
    Current Medications:
    Metformin 500mg
    Lisinopril 10mg
    """
    res = nlp.analyze_medical_report_text(report_text)

    assert res["success"] is True
    assert len(res["extracted_tests"]) >= 2
    assert res["abnormal_findings_count"] >= 1
    assert len(res["medications_detected"]) >= 1
    med_names = [m["medication"] for m in res["medications_detected"]]
    assert "Metformin" in med_names
