import streamlit as st
import pandas as pd
import os
from datetime import datetime, timedelta
from data_manager import DataManager
from modules import render_cards, render_filter

# --- Configuration ---
PAGE_TITLE = "MNS | Sentiment Validation"
PAGE_ICON = "📈"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(os.path.dirname(BASE_DIR), "mns_demo_enriched.csv")
FEEDBACK_PATH = os.path.join(BASE_DIR, "feedback_log.csv")

# --- Setup ---
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

# --- Main Execution ---
def main():
    load_css()
    manager = get_manager()
    df = manager.load_data()
    
    view_df, selected_ticker = render_filter.render_sidebar(df, manager, DATA_PATH)
    
    st.title(f"News Analysis: {selected_ticker}")
    render_cards.render_metrics(view_df, manager)
    
    for index, row in view_df.iterrows():
        render_cards.render_news_card(index, row, manager)

if __name__ == "__main__":
    main()
