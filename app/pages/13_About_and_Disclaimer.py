"""
CliNexa Healthcare Intelligence Platform
Module 13: Technical Architecture, Model Documentation & Clinical Disclaimer
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from app.components.disclaimer import render_medical_disclaimer_banner, render_footer, MANDATORY_DISCLAIMER_TEXT
from app.components.header import render_page_header

st.set_page_config(page_title="About & Disclaimer | CliNexa", page_icon="⚖️", layout="wide")

render_medical_disclaimer_banner()
render_page_header(
    title="Technical Architecture & Clinical Disclaimer",
    subtitle="Model specifications, safety guardrails, privacy protocols, and regulatory disclaimer.",
    badge="Ethics & Architecture"
)

# Mandatory Clinical Disclaimer Box
st.markdown("### 1. Mandatory Clinical Safety Disclaimer")
st.error(MANDATORY_DISCLAIMER_TEXT)

st.markdown("---")

# Architectural Model Overview
st.subheader("2. AI / ML Model Specifications")
st.markdown(
    """
    CliNexa uses modern deep learning, transformer NLP, computer vision, and explainable AI pipelines.
    
    **Architectural Exclusions:**
    > In accordance with strict design principles, this project explicitly **does NOT utilize** Logistic Regression, Random Forest, or XGBoost.
    """
)

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
        <div style="background: white; border: 1px solid #E2E8F0; border-radius: 8px; padding: 18px; margin-bottom: 14px;">
            <h4 style="margin: 0 0 6px 0; color: #0284C7;">🧬 Biomedical NLP & Clinical Transformers</h4>
            <p style="font-size: 13.5px; color: #475569; line-height: 1.5;">
                • <strong>Backbones:</strong> ClinicalBERT (Bio_ClinicalBERT) / DistilBERT.<br>
                • <strong>Function:</strong> Tokenization, named entity recognition for symptoms, laboratory tests, reference ranges, and medications.<br>
                • <strong>Guardrail:</strong> Uses strictly non-diagnostic phrasing such as 'Possible health concern' and 'Further medical evaluation may be appropriate'.
            </p>
        </div>
        
        <div style="background: white; border: 1px solid #E2E8F0; border-radius: 8px; padding: 18px; margin-bottom: 14px;">
            <h4 style="margin: 0 0 6px 0; color: #0D9488;">🫁 Medical Computer Vision & Grad-CAM</h4>
            <p style="font-size: 13.5px; color: #475569; line-height: 1.5;">
                • <strong>Backbone:</strong> ResNet-50 Deep Residual Convolutional Neural Network.<br>
                • <strong>Training:</strong> Transfer learning on thoracic radiograph datasets.<br>
                • <strong>Explainability:</strong> Gradient-weighted Class Activation Mapping (Grad-CAM) targeting <code>layer4</code> bottleneck convolutions.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        """
        <div style="background: white; border: 1px solid #E2E8F0; border-radius: 8px; padding: 18px; margin-bottom: 14px;">
            <h4 style="margin: 0 0 6px 0; color: #6366F1;">🧠 Deep Clinical Risk Classifier & SHAP</h4>
            <p style="font-size: 13.5px; color: #475569; line-height: 1.5;">
                • <strong>Architecture:</strong> Deep PyTorch Multi-Layer Perceptron (MLP) with Batch Normalization, Dropout, and Multi-Head Risk outputs.<br>
                • <strong>Explainability:</strong> SHAP (Shapley Additive Explanations) computing feature importance and directional attribution.
            </p>
        </div>
        
        <div style="background: white; border: 1px solid #E2E8F0; border-radius: 8px; padding: 18px; margin-bottom: 14px;">
            <h4 style="margin: 0 0 6px 0; color: #F59E0B;">📚 Grounded RAG (FAISS Vector Store)</h4>
            <p style="font-size: 13.5px; color: #475569; line-height: 1.5;">
                • <strong>Vector DB:</strong> FAISS (Facebook AI Similarity Search) with Inner Product cosine metric.<br>
                • <strong>Corpus:</strong> Verified clinical documentation from WHO, AHA, NIH, and CDC.<br>
                • <strong>Anti-Hallucination:</strong> Grounded responses with explicit literature citations.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")

# Privacy and Security Section
st.subheader("3. Patient Privacy & Data Protection Protocols")
st.markdown(
    """
    - **In-Memory Processing Only:** Uploaded medical documents (PDFs, images) are parsed transiently in system RAM and are never permanently stored on local or cloud disks.
    - **Zero Data Harvesting:** No patient data or health metrics are sold, retained, or utilized for unauthorized model training.
    - **HIPAA-Aligned Design:** Architecture structured to comply with de-identification and sensitive health data handling principles.
    - **Open-Source Local Execution:** CliNexa is completely capable of offline local execution without mandatory paid third-party API dependencies.
    """
)

render_footer()
