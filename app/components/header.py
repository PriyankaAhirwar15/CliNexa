"""
CliNexa Healthcare Intelligence Platform
Module: Header & Page Layout Component
Description: Professional healthcare header with branding, badges, and layout styling.
"""

import streamlit as st


def render_page_header(title: str, subtitle: str, badge: str = "Clinical Intelligence"):
    """Render consistent modern healthcare page header."""
    st.markdown(
        f"""
        <div style="margin-bottom: 20px;">
            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px;">
                <div style="display: flex; align-items: center; gap: 12px;">
                    <div style="background: linear-gradient(135deg, #0284C7, #0F766E); width: 42px; height: 42px; border-radius: 10px; display: flex; align-items: center; justify-content: center; color: white; font-size: 22px; font-weight: bold; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);">
                        ⚕️
                    </div>
                    <div>
                        <h1 style="margin: 0; padding: 0; font-size: 28px; font-weight: 700; color: #0F172A; letter-spacing: -0.5px;">
                            {title}
                        </h1>
                        <p style="margin: 3px 0 0 0; color: #475569; font-size: 14.5px;">
                            {subtitle}
                        </p>
                    </div>
                </div>
                <span style="background-color: #E0F2FE; color: #0369A1; font-size: 12px; font-weight: 600; padding: 5px 12px; border-radius: 9999px; border: 1px solid #BAE6FD;">
                    {badge}
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
