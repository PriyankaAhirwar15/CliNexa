"""
CliNexa Healthcare Intelligence Platform
Module 3: Medical Text & Document Analysis
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
from app.utils.document_parser import extract_text_from_file
from models.nlp.transformer_nlp import BiomedicalNLPService

st.set_page_config(page_title="Medical Text Analysis | CliNexa", page_icon="📄", layout="wide")

render_medical_disclaimer_banner()
render_page_header(
    title="Module 3: Medical Text & Clinical Report Analysis",
    subtitle="Ingest clinical reports (PDF, DOCX, TXT) and extract laboratory metrics, abnormal values, and prescribed medications.",
    badge="Clinical Report Parser"
)

if "nlp_service" not in st.session_state:
    st.session_state.nlp_service = BiomedicalNLPService()

nlp_service = st.session_state.nlp_service

# Sample Reports Quick Loaders
sample_dir = PROJECT_ROOT / "data" / "sample" / "sample_reports"
st.markdown("##### Quick-Load Clinical Case Studies:")
qc1, qc2, qc3 = st.columns(3)

loaded_text = ""
with qc1:
    if st.button("Load: Comprehensive Metabolic Panel (TXT)", use_container_width=True):
        sample_path = sample_dir / "routine_metabolic_panel.txt"
        if sample_path.exists():
            loaded_text = sample_path.read_text(encoding="utf-8")
with qc2:
    if st.button("Load: Outpatient Cardiovascular Follow-Up (TXT)", use_container_width=True):
        sample_path = sample_dir / "cardiovascular_assessment.txt"
        if sample_path.exists():
            loaded_text = sample_path.read_text(encoding="utf-8")
with qc3:
    if st.button("Clear Input Area", use_container_width=True):
        loaded_text = ""

# Document Upload Section
uploaded_file = st.file_uploader(
    "Or upload a clinical report document (Supported: PDF, DOCX, TXT - Max 10MB):",
    type=["pdf", "docx", "txt"]
)

raw_document_text = ""
if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    parse_result = extract_text_from_file(file_bytes, uploaded_file.name)
    if parse_result["success"]:
        raw_document_text = parse_result["text"]
        st.success(f"Successfully extracted text from '{uploaded_file.name}' ({len(raw_document_text)} characters).")
    else:
        st.error(parse_result["error"])
elif loaded_text:
    raw_document_text = loaded_text

# Text Area for manual edit or review
final_text = st.text_area(
    "Clinical Document Content:",
    value=raw_document_text,
    height=200,
    placeholder="Paste or review clinical document text here..."
)

analyze_doc_btn = st.button("Analyze Medical Document", type="primary", use_container_width=False)

if analyze_doc_btn or (uploaded_file is not None and raw_document_text):
    if not final_text.strip():
        st.warning("Please provide clinical document text or upload a supported file.")
    else:
        with st.spinner("Extracting clinical entities, laboratory tests, and medications..."):
            res = nlp_service.analyze_medical_report_text(final_text)

        if not res["success"]:
            st.error(res["error"])
        else:
            st.session_state.latest_report_analysis = res
            st.success("Clinical document analyzed successfully.")

            # Metric Summary Cards
            abnormal_count = res["abnormal_findings_count"]
            total_tests = len(res["extracted_tests"])
            meds_count = len(res["medications_detected"])

            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                render_metric_card(
                    title="Identified Lab Tests",
                    value=f"{total_tests} Tests",
                    subtitle="Extracted via Biomedical NER",
                    color="#0284C7"
                )
            with sc2:
                render_metric_card(
                    title="Out-of-Range Markers",
                    value=f"{abnormal_count} Values",
                    subtitle="Compared to standard reference limits",
                    color="#EF4444" if abnormal_count > 0 else "#10B981"
                )
            with sc3:
                render_metric_card(
                    title="Documented Medications",
                    value=f"{meds_count} Drugs",
                    subtitle="Pharmacological classes identified",
                    color="#F59E0B"
                )

            st.markdown("<br>", unsafe_allow_html=True)

            # 1. Laboratory Findings
            st.subheader("1. Laboratory Panels & Value Status")
            if res["extracted_tests"]:
                tests_display = []
                for t in res["extracted_tests"]:
                    tests_display.append({
                        "Laboratory Test": t["test_name"],
                        "Measured Value": f"{t['measured_value']} {t['unit']}",
                        "Standard Reference Range": t["reference_range"],
                        "Status Assessment": t["status"],
                        "Physiological System": t["health_category"]
                    })
                st.dataframe(tests_display, use_container_width=True)
            else:
                st.info("No standard laboratory numeric tests detected in the text.")

            # 2. Pharmacotherapy
            st.subheader("2. Detected Pharmacotherapy & Prescriptions")
            if res["medications_detected"]:
                meds_display = []
                for m in res["medications_detected"]:
                    meds_display.append({
                        "Medication Name": m["medication"],
                        "Dosage Mentioned": m["dosage"],
                        "Drug Classification": m["drug_class"],
                        "Primary Clinical Indication": m["primary_indication"]
                    })
                st.table(meds_display)
            else:
                st.info("No prescribed medications explicitly identified in this document.")

            # 3. Clinical Observations
            st.subheader("3. Clinical Observations & Educational Synthesis")
            st.markdown(
                """
                <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 18px;">
                """,
                unsafe_allow_html=True
            )
            for obs in res["clinical_observations"]:
                st.markdown(f"• **Observation:** {obs}")

            st.markdown(
                f"""
                    <hr style="margin: 12px 0; border: none; border-top: 1px solid #E2E8F0;">
                    <p style="margin: 0; color: #64748B; font-size: 13px;">
                        <em>Disclaimer:</em> {res['disclaimer']}
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

render_footer()
