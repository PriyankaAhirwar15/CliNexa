"""
CliNexa Healthcare Intelligence Platform
Module: Session State Manager
Description: Centralized state management for user health profile,
nutrition parameters, symptom records, and analysis history across Streamlit pages.
"""

import streamlit as st
from typing import Dict, Any


DEFAULT_PROFILE = {
    "age": 38,
    "gender": "Male",
    "height_cm": 175.0,
    "weight_kg": 76.0,
    "activity_level": "Moderately active",
    "dietary_preference": "Vegetarian",
    "food_allergies": [],
    "health_conditions": ["Mild Hypertension history"],
    "current_medications": ["None currently prescribed"],
    "smoking_status": "Non-smoker",
    "alcohol_consumption": "Occasional / Light",
    "sleep_duration": 7.0,
    "daily_water_intake": 2.5,
    "systolic_bp": 124,
    "diastolic_bp": 82
}


def init_session_state():
    """Initialize default session state keys if not already set."""
    if "user_profile" not in st.session_state:
        st.session_state.user_profile = DEFAULT_PROFILE.copy()

    if "latest_symptom_analysis" not in st.session_state:
        st.session_state.latest_symptom_analysis = None

    if "latest_report_analysis" not in st.session_state:
        st.session_state.latest_report_analysis = None

    if "latest_vision_analysis" not in st.session_state:
        st.session_state.latest_vision_analysis = None

    if "rag_chat_history" not in st.session_state:
        st.session_state.rag_chat_history = []


def get_user_profile() -> Dict[str, Any]:
    """Retrieve current user profile from session."""
    init_session_state()
    return st.session_state.user_profile


def update_user_profile(new_data: Dict[str, Any]):
    """Update user profile keys in session."""
    init_session_state()
    st.session_state.user_profile.update(new_data)
