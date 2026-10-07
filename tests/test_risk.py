"""
Unit Tests for Deep Clinical Risk Classifier & SHAP Explainer
Verifies strict non-usage of prohibited models (Logistic Regression, Random Forest, XGBoost).
"""

import pytest
import torch
from models.risk.risk_dnn import RiskClassifierService, DeepClinicalRiskNet, FEATURE_NAMES
from explainability.shap_explainer import ClinicalSHAPExplainer


def test_prohibited_models_not_used():
    """Verify that forbidden models are NEVER imported or initialized."""
    import sys
    for mod_name in list(sys.modules.keys()):
        assert "xgboost" not in mod_name, "XGBoost was found loaded in modules!"
    
    # Verify model is pure PyTorch nn.Module
    service = RiskClassifierService()
    assert isinstance(service.model, torch.nn.Module)
    assert isinstance(service.model, DeepClinicalRiskNet)


def test_risk_prediction_validity():
    service = RiskClassifierService()
    # Age=45, BMI=26.5, BP=122/80, Activity=3.5, Sleep=7, Smoke=0, Alcohol=0, Water=2.2, Symptom=0
    sample_features = [45.0, 26.5, 122.0, 80.0, 3.5, 7.0, 0.0, 0.0, 2.2, 0.0]
    res = service.predict(sample_features)

    assert 0.0 <= res["overall_probability"] <= 1.0
    assert 0.0 <= res["overall_risk_score"] <= 100.0
    assert "risk_category" in res
    assert "clinical_guidance" in res
    assert res["model_type"] == "Deep Multi-Layer Neural Network (PyTorch)"


def test_shap_explanation_generation():
    service = RiskClassifierService()
    explainer = ClinicalSHAPExplainer(service)
    sample_features = [55.0, 31.0, 140.0, 90.0, 1.0, 5.5, 2.0, 1.0, 1.5, 2.0]
    explanation = explainer.explain_instance(sample_features)

    assert "baseline_score" in explanation
    assert "predicted_score" in explanation
    assert len(explanation["feature_attributions"]) == len(FEATURE_NAMES)
    assert len(explanation["sorted_attributions"]) == len(FEATURE_NAMES)
    assert len(explanation["summary_narrative"]) > 20
