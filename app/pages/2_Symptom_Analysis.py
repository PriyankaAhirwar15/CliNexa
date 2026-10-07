"""
CliNexa Healthcare Intelligence Platform
Module 2: Natural Language Symptom Analysis
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from app.components.disclaimer import render_medical_disclaimer_banner, render_footer
from app.components.header import render_page_header
from app.components.cards import render_metric_card
from models.nlp.transformer_nlp import BiomedicalNLPService

st.set_page_config(page_title="Symptom Analysis | CliNexa", page_icon="💬", layout="wide")

render_medical_disclaimer_banner()
render_page_header(
    title="Module 2: Natural Language Symptom Analysis",
    subtitle="Extract medical concepts, symptom duration, severity markers, and potential health categories using biomedical transformers.",
    badge="Biomedical NLP (ClinicalBERT)"
)

# Initialize NLP Service
if "nlp_service" not in st.session_state:
    st.session_state.nlp_service = BiomedicalNLPService()

nlp_service = st.session_state.nlp_service

# Sample Scenarios for 1-Click Exploration
st.markdown("##### Quick Demonstration Scenarios:")
quick_col1, quick_col2, quick_col3 = st.columns(3)

sample_input = ""
with quick_col1:
    if st.button("Scenario A: Metabolic Symptoms", use_container_width=True):
        sample_input = "I have been experiencing ongoing fatigue, persistent thirst, and frequent urination for the past 3 weeks."
with quick_col2:
    if st.button("Scenario B: Cardiopulmonary Discomfort", use_container_width=True):
        sample_input = "Occasional mild chest pressure and slight shortness of breath when walking up hills for several days."
with quick_col3:
    if st.button("Scenario C: Musculoskeletal & Headaches", use_container_width=True):
        sample_input = "Moderate joint pain in both knees and recurring tension headache during the past 10 days."

symptom_text = st.text_area(
    "Describe what you have been experiencing in plain English:",
    value=sample_input if sample_input else "",
    placeholder="Example: I have been experiencing persistent fatigue, increased thirst, and frequent urination for the past 3 weeks...",
    height=130
)

col_run, col_clear = st.columns([1, 5])
with col_run:
    analyze_btn = st.button("Analyze Symptoms", type="primary", use_container_width=True)

if analyze_btn or sample_input:
    if not symptom_text.strip():
        st.warning("Please provide a description of your symptoms before running analysis.")
    else:
        with st.spinner("Processing text with biomedical transformer & clinical concept ontologies..."):
            result = nlp_service.analyze_symptoms(symptom_text)

        if not result["success"]:
            st.error(result.get("error", "Analysis could not be completed."))
        else:
            st.session_state.latest_symptom_analysis = result
            st.success("Symptom analysis completed successfully.")

            # Summary Indicator Cards
            c1, c2, c3 = st.columns(3)
            with c1:
                render_metric_card(
                    title="Estimated Duration",
                    value=result["duration"],
                    subtitle="Extracted temporal marker",
                    color="#0284C7"
                )
            with c2:
                render_metric_card(
                    title="Severity Indicator",
                    value=result["severity_indicator"],
                    subtitle="Natural language intensity marker",
                    color="#D97706" if "Severe" in result["severity_indicator"] else "#10B981"
                )
            with c3:
                urgency_badge = "Further Medical Evaluation Advised" if result["has_urgent_signals"] else "Routine Review Appropriate"
                render_metric_card(
                    title="Clinical Decision Support",
                    value=urgency_badge,
                    subtitle="Non-diagnostic safety guidance",
                    color="#DC2626" if result["has_urgent_signals"] else "#2563EB"
                )

            st.markdown("<br>", unsafe_allow_html=True)

            # Identified Clinical Entities
            st.subheader("1. Extracted Biomedical Findings")
            if result["identified_symptoms"]:
                table_data = []
                for s in result["identified_symptoms"]:
                    table_data.append({
                        "Reported Term": s["patient_term"],
                        "Standard Clinical Nomenclature": s["clinical_nomenclature"],
                        "Relevant Health Category": s["health_category"],
                        "Clinical Assessment Level": s["evaluation_level"]
                    })
                st.table(table_data)
            else:
                st.info("No standardized high-confidence clinical symptoms matched from the vocabulary dictionary. General health concepts were extracted.")

            # Categories & Safe Guidance
            st.subheader("2. Health Categorization & Safe Advisory")
            st.markdown(
                f"""
                <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 20px;">
                    <p style="margin: 0 0 10px 0; font-size: 15px;">
                        <strong>Relevant Health Categories:</strong> {', '.join(result['relevant_categories'])}
                    </p>
                    <p style="margin: 0 0 10px 0; font-size: 15px; color: #1E293B;">
                        <strong>Potential Risk Indicator:</strong> {result['potential_risk_indicator']}
                    </p>
                    <p style="margin: 0 0 10px 0; font-size: 15px; color: #0369A1;">
                        <strong>Clinical Recommendation:</strong> {result['recommended_action']}
                    </p>
                    <hr style="margin: 14px 0; border: none; border-top: 1px solid #E2E8F0;">
                    <p style="margin: 0; font-size: 13px; color: #64748B; font-style: italic;">
                        <strong>Safe Communication Commitment:</strong> CliNexa does not declare definitive disease diagnoses (e.g. 'You have diabetes'). Rather, it identifies potential clinical patterns to empower informed, constructive dialogues with qualified medical practitioners.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

render_footer()
