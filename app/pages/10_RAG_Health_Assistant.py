"""
CliNexa Healthcare Intelligence Platform
Module 7: RAG Health Information Assistant
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
from rag.retriever import ClinicalRAGSystem

st.set_page_config(page_title="RAG Health Assistant | CliNexa", page_icon="🧠", layout="wide")

render_medical_disclaimer_banner()
render_page_header(
    title="Module 7: Grounded RAG Healthcare Assistant",
    subtitle="Retrieval-Augmented Generation grounded in verified clinical guidelines from WHO, AHA, NIH, and CDC.",
    badge="Vector Retrieval (FAISS)"
)

# Initialize RAG System
if "rag_system" not in st.session_state:
    with st.spinner("Initializing FAISS vector index over clinical literature..."):
        st.session_state.rag_system = ClinicalRAGSystem()

rag_system = st.session_state.rag_system

# Quick Exploration Queries
st.markdown("##### Recommended Healthcare Queries:")
qc1, qc2, qc3, qc4 = st.columns(4)

prefill_q = ""
with qc1:
    if st.button("Role of Dietary Fiber?", use_container_width=True):
        prefill_q = "What is the general physiological role of dietary fiber?"
with qc2:
    if st.button("Hypertension Lifestyle Factors?", use_container_width=True):
        prefill_q = "What are common lifestyle factors associated with high blood pressure?"
with qc3:
    if st.button("HbA1c & Fasting Glucose?", use_container_width=True):
        prefill_q = "What are the standard diagnostic reference ranges for HbA1c and fasting blood sugar?"
with qc4:
    if st.button("AHA Life's Essential 8?", use_container_width=True):
        prefill_q = "What key risk factors and lifestyle behaviors does the American Heart Association emphasize?"

# Input Box
query_input = st.text_input(
    "Ask a clinical or nutritional health question in English:",
    value=prefill_q if prefill_q else "",
    placeholder="e.g. What does elevated serum creatinine indicate? or How does potassium affect blood pressure?"
)

ask_btn = st.button("Submit Query", type="primary")

if ask_btn or prefill_q:
    if not query_input.strip():
        st.warning("Please type a question before submitting.")
    else:
        with st.spinner("Executing semantic vector retrieval in FAISS and synthesizing grounded response..."):
            res = rag_system.answer_query(query_input)

        if not res["success"]:
            st.error(res["error"])
        else:
            # Display Grounded Answer
            st.markdown("### Grounded Clinical Response")
            st.markdown(
                f"""
                <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: 22px; margin-bottom: 20px;">
                    <div style="font-size: 15px; line-height: 1.65; color: #1E293B;">
                        {res['answer']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Display Source Literature Citations
            st.subheader("Retrieved Reference Literature & Citations")
            for idx, src in enumerate(res["sources"]):
                with st.expander(f"Source [{idx+1}]: {src['document_title']} (Semantic Match Score: {src['score']:.4f})"):
                    st.markdown(f"**Document File:** `{src['document_name']}`")
                    st.markdown(f"**Indexed Text Segment:**\n\n>{src['content']}")

            # Safety statement
            st.markdown(
                f"""
                <div style="background-color: #EFF6FF; border-left: 4px solid #3B82F6; padding: 12px 16px; border-radius: 4px; margin-top: 15px;">
                    <p style="margin: 0; color: #1E40AF; font-size: 13px;">
                        🛡️ <strong>Safety & Anti-Hallucination Guardrail:</strong> {res['safety_disclaimer']}
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

render_footer()
