"""
CliNexa Healthcare Intelligence Platform
Module 4 & 5: Medical Image Analysis & Grad-CAM Visual Explainability
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from PIL import Image
import plotly.graph_objects as go
from app.components.disclaimer import render_medical_disclaimer_banner, render_footer
from app.components.header import render_page_header
from app.components.cards import render_metric_card
from models.vision.resnet_classifier import MedicalVisionService
from explainability.gradcam import GradCAMExplainer

st.set_page_config(page_title="Medical Image Analysis | CliNexa", page_icon="🫁", layout="wide")

render_medical_disclaimer_banner()
render_page_header(
    title="Module 4 & 5: Medical Image Analysis & Grad-CAM Explainability",
    subtitle="Deep convolutional transfer learning with ResNet-50 and gradient-weighted class activation mapping (Grad-CAM).",
    badge="Computer Vision (ResNet-50 + Grad-CAM)"
)

# Initialize Vision & Grad-CAM Services
if "vision_service" not in st.session_state:
    st.session_state.vision_service = MedicalVisionService()

vision_service = st.session_state.vision_service
gradcam_explainer = GradCAMExplainer(vision_service)

# Sample images directory
sample_img_dir = PROJECT_ROOT / "data" / "sample" / "sample_images"

st.markdown("##### Quick-Load Demonstration Chest Radiographs:")
sc1, sc2, sc3 = st.columns(3)

chosen_sample_path = None
with sc1:
    if st.button("Load: Normal Thoracic Study (Chest X-Ray)", use_container_width=True):
        chosen_sample_path = sample_img_dir / "normal_chest_xray.png"
with sc2:
    if st.button("Load: Pneumonia Infiltration Study (Chest X-Ray)", use_container_width=True):
        chosen_sample_path = sample_img_dir / "pneumonia_chest_xray.png"
with sc3:
    if st.button("Load: Cardiomegaly Study (Enlarged Heart)", use_container_width=True):
        chosen_sample_path = sample_img_dir / "cardiomegaly_chest_xray.png"

# File uploader
uploaded_img = st.file_uploader(
    "Or upload a medical radiograph image (PNG, JPG, JPEG):",
    type=["png", "jpg", "jpeg"]
)

active_image = None
if uploaded_img is not None:
    try:
        active_image = Image.open(uploaded_img)
    except Exception as e:
        st.error(f"Error reading image: {str(e)}")
elif chosen_sample_path and chosen_sample_path.exists():
    active_image = Image.open(chosen_sample_path)

if active_image is not None:
    st.markdown("---")
    blend_alpha = st.slider("Grad-CAM Overlay Heatmap Transparency (Alpha):", min_value=0.2, max_value=0.8, value=0.5, step=0.05)

    with st.spinner("Executing ResNet-50 forward pass and Grad-CAM backpropagation on layer4..."):
        pred_res = vision_service.predict(active_image)
        gradcam_res = gradcam_explainer.generate_visualizations(active_image, alpha=blend_alpha)
        st.session_state.latest_vision_analysis = pred_res

    # Metric Cards
    m1, m2, m3 = st.columns(3)
    with m1:
        render_metric_card(
            title="Predicted Classification",
            value=pred_res["predicted_class"],
            subtitle="Deep ResNet-50 Transfer Learning",
            color="#0284C7"
        )
    with m2:
        render_metric_card(
            title="Model Confidence",
            value=f"{pred_res['confidence_percent']}%",
            subtitle="Softmax output probability",
            color="#10B981" if pred_res["confidence_percent"] > 60 else "#F59E0B"
        )
    with m3:
        render_metric_card(
            title="Model Architecture",
            value="ResNet-50 CNN",
            subtitle="layer4 bottleneck convolutional target",
            color="#6366F1"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # 3-Panel Visual Comparison
    st.subheader("Grad-CAM Visual Heatmap & Attention Inspection")
    img_col1, img_col2, img_col3 = st.columns(3)

    with img_col1:
        st.markdown("<h5 style='text-align: center; color: #1E293B;'>1. Original Medical Radiograph</h5>", unsafe_allow_html=True)
        st.image(gradcam_res["original_image"], use_container_width=True)

    with img_col2:
        st.markdown("<h5 style='text-align: center; color: #1E293B;'>2. Grad-CAM Activation Heatmap</h5>", unsafe_allow_html=True)
        st.image(gradcam_res["heatmap_image"], use_container_width=True)

    with img_col3:
        st.markdown("<h5 style='text-align: center; color: #1E293B;'>3. Superimposed Heatmap Overlay</h5>", unsafe_allow_html=True)
        st.image(gradcam_res["overlay_image"], use_container_width=True)

    # Required Explanation Notice
    st.markdown(
        f"""
        <div style="background-color: #FEF3C7; border-left: 5px solid #F59E0B; padding: 16px; border-radius: 6px; margin: 18px 0;">
            <p style="margin: 0; font-size: 14.5px; color: #92400E; font-weight: 600;">
                {gradcam_res['explanation_notice']}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Probability Distribution Plot
    st.subheader("Model Class Probability Distribution")
    probs = pred_res["probabilities"]
    classes = [p["class_name"] for p in probs]
    confidences = [p["probability"] for p in probs]

    fig = go.Figure(go.Bar(
        x=confidences,
        y=classes,
        orientation="h",
        marker=dict(color="#0284C7"),
        text=[f"{c:.1f}%" for c in confidences],
        textposition="outside"
    ))
    fig.update_layout(
        title="<b>Class Probabilities across Thoracic Diagnostic Categories</b>",
        xaxis_title="Confidence Percentage (%)",
        yaxis_autorange="reversed",
        font=dict(family="Arial, sans-serif", size=13),
        height=300,
        margin=dict(l=220, r=40, t=40, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Technical Model Card
    with st.expander("Technical Model Card & Demonstration Limitations"):
        st.markdown(
            f"""
            - **Backbone Architecture:** ResNet-50 (Deep Residual Convolutional Neural Network with 25.6M parameters).
            - **Feature Extraction:** Pretrained ImageNet convolutional layers freezing blocks 1 through 3, fine-tuning block 4.
            - **Attention Hook:** `model.backbone.layer4[-1]` gradient tracking with ReLU rectification.
            - **Demonstration Classification Envelope:** Educational proof-of-concept. The system does NOT represent commercial radiology software and is NOT approved for clinical diagnostic workflows.
            """
        )
else:
    st.info("Select a demonstration radiograph above or upload a chest X-ray image to begin.")

render_footer()
