"""
CliNexa Healthcare Intelligence Platform
Module: UI Card Components
Description: Reusable cards for clinical metrics, risk badges, and recommendations.
"""

import streamlit as st
from typing import Dict, Any, List


def render_metric_card(title: str, value: str, subtitle: str = "", delta: str = None, color: str = "#0284C7"):
    """Render a clean healthcare metric card."""
    delta_html = f"<div style='color: {color}; font-size: 12px; font-weight: 600; margin-top: 4px;'>{delta}</div>" if delta else ""
    st.markdown(
        f"""
        <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px; padding: 18px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); height: 100%;">
            <div style="font-size: 13px; font-weight: 600; color: #64748B; text-transform: uppercase; letter-spacing: 0.5px;">
                {title}
            </div>
            <div style="font-size: 26px; font-weight: 700; color: #0F172A; margin: 6px 0 2px 0;">
                {value}
            </div>
            <div style="font-size: 12.5px; color: #64748B;">
                {subtitle}
            </div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True
    )


def render_risk_indicator_box(category: str, score: float, guidance: str, badge_color: str = "blue"):
    """Render a prominent healthcare risk assessment box."""
    color_map = {
        "green": {"bg": "#ECFDF5", "border": "#10B981", "text": "#065F46", "badge": "#059669"},
        "blue": {"bg": "#EFF6FF", "border": "#3B82F6", "text": "#1E40AF", "badge": "#2563EB"},
        "orange": {"bg": "#FFFBEB", "border": "#F59E0B", "text": "#92400E", "badge": "#D97706"},
        "red": {"bg": "#FEF2F2", "border": "#EF4444", "text": "#991B1B", "badge": "#DC2626"}
    }
    cfg = color_map.get(badge_color, color_map["blue"])

    st.markdown(
        f"""
        <div style="background-color: {cfg['bg']}; border-left: 6px solid {cfg['border']}; border-radius: 8px; padding: 20px; margin: 15px 0 20px 0;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-size: 18px; font-weight: 700; color: {cfg['text']};">
                    {category}
                </span>
                <span style="background-color: {cfg['badge']}; color: white; padding: 4px 12px; border-radius: 9999px; font-size: 13px; font-weight: 600;">
                    Model Estimate: {score}%
                </span>
            </div>
            <div style="color: {cfg['text']}; font-size: 14px; line-height: 1.55;">
                <strong>Clinical Decision Support Note:</strong> {guidance}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
