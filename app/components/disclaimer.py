"""
CliNexa Healthcare Intelligence Platform
Module: Medical Disclaimer Component
Description: Standardized healthcare safety advisory banner for consistent display
across all application modules.
"""

import streamlit as st

MANDATORY_DISCLAIMER_TEXT = (
    "This application provides AI-assisted health information and general nutrition "
    "guidance for educational purposes only. It is not a medical diagnosis or a substitute "
    "for professional medical advice. Please consult a qualified healthcare professional "
    "for diagnosis and treatment."
)


def render_medical_disclaimer_banner():
    """Render the standard regulatory and clinical disclaimer banner."""
    st.markdown(
        f"""
        <div style="background-color: #FEF3C7; border-left: 5px solid #F59E0B; padding: 12px 18px; border-radius: 6px; margin-bottom: 22px;">
            <div style="display: flex; align-items: flex-start;">
                <div style="font-size: 20px; margin-right: 12px; color: #B45309; line-height: 1;">⚠️</div>
                <div style="color: #92400E; font-size: 13.5px; line-height: 1.5; font-weight: 500;">
                    <strong>IMPORTANT HEALTHCARE & CLINICAL NOTICE:</strong><br>
                    {MANDATORY_DISCLAIMER_TEXT}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_footer():
    """Render footer with disclaimer notice and branding."""
    st.markdown("---")
    st.markdown(
        f"""
        <div style="text-align: center; color: #64748B; font-size: 12.5px; padding: 10px 0 20px 0;">
            <p><strong>CliNexa Healthcare Intelligence Platform</strong> • AI-Assisted Clinical Decision Support</p>
            <p style="max-width: 820px; margin: 0 auto; line-height: 1.4;">{MANDATORY_DISCLAIMER_TEXT}</p>
            <p style="margin-top: 8px; font-size: 11.5px; color: #94A3B8;">Built for educational, academic, and research demonstration purposes only.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
