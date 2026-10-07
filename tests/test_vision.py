"""
Unit Tests for ResNet-50 Medical Vision Model & Grad-CAM Explainer
"""

import pytest
import numpy as np
from PIL import Image
from models.vision.resnet_classifier import MedicalVisionService, CLASSES
from explainability.gradcam import GradCAMExplainer


def test_resnet_vision_prediction():
    service = MedicalVisionService()
    # Create test image (512x512 RGB)
    arr = np.random.randint(50, 200, (512, 512, 3), dtype=np.uint8)
    img = Image.fromarray(arr)

    res = service.predict(img)
    assert res["predicted_class"] in CLASSES
    assert 0.0 <= res["confidence_percent"] <= 100.0
    assert len(res["probabilities"]) == len(CLASSES)
    assert res["is_demonstration"] is True


def test_gradcam_visualizations():
    service = MedicalVisionService()
    explainer = GradCAMExplainer(service)

    arr = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
    img = Image.fromarray(arr)

    vis = explainer.generate_visualizations(img, alpha=0.5)
    assert "original_image" in vis
    assert "heatmap_image" in vis
    assert "overlay_image" in vis
    assert vis["original_image"].size == vis["overlay_image"].size
    assert "explanation_notice" in vis
    assert "Highlighted regions" in vis["explanation_notice"]
