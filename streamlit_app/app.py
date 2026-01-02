import streamlit as st
import pandas as pd
import os
from datetime import datetime, timedelta
from data_manager import DataManager
from modules import render_cards, render_filters, feedback

# --- Configuration ---
PAGE_TITLE = "MNS | Sentiment Validation"
PAGE_ICON = "📈"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(os.path.dirname(BASE_DIR), "mns_demo_enriched.csv")
FEEDBACK_PATH = os.path.join(BASE_DIR, "feedback_log.csv")

# --- PageSetup ---
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Helper Functions ---
def load_css():
    """Loads the custom CSS."""
    css_path = os.path.join(os.path.dirname(__file__), 'assets', 'style.css')
    with open(css_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True) #allow injection of customized html 

@st.cache_resource
def get_manager():
    """Initializes and caches the DataManager."""
    return DataManager(DATA_PATH, FEEDBACK_PATH)

@st.cache_data  # Cache indefinitely (max time) - only clears on app restart
def load_data_cached(_manager):
    """
    Loads and caches data from BigQuery or CSV.
    The underscore prefix on _manager prevents Streamlit from hashing it.
    Data stays cached until app restart or manual cache clear.
    """
    return _manager.load_data()

def load_user_guide():
        
    # --- Help Button ---
    try:
        # Resolve PDF Path corresponding to vgeo-mns/MNS_Quick_User_Guide.pdf
        # This file is in vgeo-mns/streamlit_app/app.py -> dirname -> streamlit_app -> dirname -> vgeo-mns
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        pdf_path = os.path.join(base_dir, "MNS_Quick_User_Guide.pdf")
        
        if os.path.exists(pdf_path):
            # Add some vertical padding to align with the title
            st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
            with open(pdf_path, "rb") as f:
                st.download_button(
                    label="📥 User Guide",
                    data=f,
                    file_name="MNS_Quick_User_Guide.pdf",
                    mime="application/pdf",
                    help="Download the Quick User Guide"
                )
    except Exception as e:
        print(f"Error loading user guide: {e}")
        pass

    return

# --- Main Execution ---
def main():
    # Load CSS and initialize DataManager
    load_css()
    manager = get_manager()

    # Load data using cached function (queries BigQuery once, then uses cache)
    raw_df = load_data_cached(manager)
    
    # 2. Layer on Feedback (Recap Linkages, etc.)
    feedback_df = manager.load_feedback_df()
    df, recap_tree = feedback.apply_feedback_overrides(raw_df.copy(), feedback_df)

    # Render the sidebar and get the selected ticker (filtered data)
    view_df, selected_ticker = render_filters.render_sidebar(df, manager, DATA_PATH)

    # Header row: Title + User Guide
    col_title, col_guide = st.columns([6, 1])

    with col_title:
        st.title(f"News Analysis: {selected_ticker}")

    with col_guide:
        load_user_guide()

    # Render top metrics - using view_df for accuracy
    render_cards.render_metrics(view_df, manager)
    
    # Render the news cards
    for index, row in view_df.iterrows():
        # Pass the recap_tree to the card renderer so the 'Mother' knows about her 'Children'
        children_ids = recap_tree.get(index, [])
        render_cards.render_news_card(index, row, manager, children_recaps=children_ids)

if __name__ == "__main__":
    main()
