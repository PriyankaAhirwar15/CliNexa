"""
CliNexa Healthcare Intelligence Platform
Module 10: Smart Food Swap Engine
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
from nutrition.food_swap import get_categories, search_swaps, get_all_swaps

st.set_page_config(page_title="Smart Food Swap | CliNexa", page_icon="🔄", layout="wide")

render_medical_disclaimer_banner()
render_page_header(
    title="Module 10: Smart Food Swap Assistant",
    subtitle="Evidence-based dietary substitutions that reduce refined sugars, unhealthy fats, and sodium while upgrading nutrient density.",
    badge="Micro-Habit Transformation"
)

# Search & Filter
col_search, col_cat = st.columns([2, 1])
with col_search:
    search_q = st.text_input("Search food item or keyword:", placeholder="e.g. soda, chips, nuggets, rice, mayo")
with col_cat:
    categories = ["All"] + get_categories()
    selected_cat = st.selectbox("Filter by Food Category:", options=categories)

results = search_swaps(query=search_q, category=selected_cat)

st.markdown(f"**Showing {len(results)} Evidence-Based Food Substitutions:**")

# Interactive Swapper
for item in results:
    with st.container():
        st.markdown(
            f"""
            <div style="background: white; border: 1px solid #E2E8F0; border-radius: 10px; padding: 20px; margin-bottom: 18px; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; border-bottom: 1px solid #F1F5F9; padding-bottom: 8px;">
                    <span style="font-size: 13px; font-weight: 700; color: #64748B; text-transform: uppercase;">
                        Category: {item['category']}
                    </span>
                    <span style="background-color: #ECFDF5; color: #059669; font-weight: 700; font-size: 13px; padding: 3px 12px; border-radius: 9999px;">
                        ⚡ Calorie Savings: ~{item['calorie_savings']} kcal
                    </span>
                </div>
                
                <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px; margin-bottom: 14px;">
                    <div style="flex: 1; min-width: 200px; background: #FEF2F2; border-radius: 8px; padding: 12px; border: 1px solid #FECACA;">
                        <div style="font-size: 12px; color: #991B1B; font-weight: 600;">ORIGINAL ITEM:</div>
                        <div style="font-size: 16px; font-weight: 700; color: #7F1D1D; margin: 3px 0;">{item['original']}</div>
                        <div style="font-size: 12.5px; color: #991B1B;">Approx. {item['calories_original']} kcal</div>
                    </div>
                    
                    <div style="font-size: 26px; color: #0284C7; font-weight: bold; text-align: center;">
                        ➔
                    </div>
                    
                    <div style="flex: 1; min-width: 200px; background: #F0FDF4; border-radius: 8px; padding: 12px; border: 1px solid #BBF7D0;">
                        <div style="font-size: 12px; color: #166534; font-weight: 600;">HEALTHIER ALTERNATIVE:</div>
                        <div style="font-size: 16px; font-weight: 700; color: #14532D; margin: 3px 0;">{item['alternative']}</div>
                        <div style="font-size: 12.5px; color: #166534;">Approx. {item['calories_swap']} kcal</div>
                    </div>
                </div>
                
                <div style="background: #F8FAFC; border-radius: 6px; padding: 12px; font-size: 13.5px; line-height: 1.5; color: #334155;">
                    <p style="margin: 0 0 6px 0;"><strong>Nutrient Advantage:</strong> {item['macros_benefit']}</p>
                    <p style="margin: 0; color: #0F766E;"><strong>Scientific Rationale:</strong> {item['scientific_rationale']}</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

render_footer()
