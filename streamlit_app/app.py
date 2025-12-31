import streamlit as st
import pandas as pd
import os
from datetime import datetime, timedelta
from data_manager import DataManager
from modules import render_cards, render_filters

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
    df = raw_df.copy()
    
    # Store recap relationships: {original_id: [list_of_recap_ids]}
    recap_tree = {} 
    # Store user overrides: {news_id: {'recap_status': 'RECAP', 'sentiment': 0.5}}
    user_overrides = {}

    if not feedback_df.empty and 'news_id' in feedback_df.columns:
        # Get latest feedback per story
        latest_fb = feedback_df.sort_values('timestamp').groupby('news_id').tail(1)
        
        for _, fb in latest_fb.iterrows():
            nid = fb['news_id']
            user_overrides[nid] = {
                'is_recap': fb.get('is_recap', False),
                'linked_original_id': fb.get('linked_original_id')
            }
            
            # If this is linked to an original, add it to the tree
            orig_id = fb.get('linked_original_id')
            if pd.notna(orig_id) and orig_id != "None":
                orig_id = int(float(orig_id))
                if orig_id not in recap_tree: recap_tree[orig_id] = []
                recap_tree[orig_id].append(nid)

    # Apply overrides to the dataframe
    def apply_overrides(row):
        oid = row.name
        if oid in user_overrides:
            info = user_overrides[oid]
            if info['is_recap']:
                row['breaking_recap'] = 'Recap'
            row['linked_original_id'] = info['linked_original_id']
        else:
            row['linked_original_id'] = None
        
        # --- Auto-Boost Significance ---
        # If this story has children recaps, it's definitely significant!
        if oid in recap_tree and len(recap_tree[oid]) > 0:
            row['classification'] = 'SIGNIFICANT'
            
        return row

    df = df.apply(apply_overrides, axis=1)

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
