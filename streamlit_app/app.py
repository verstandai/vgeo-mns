import streamlit as st
import pandas as pd
import os
from datetime import datetime, timedelta
from data_manager import DataManager
from modules import render_ui

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

def render_sidebar(df, manager):
    """Renders the sidebar filters and returns the filtered dataframe and selected ticker."""
    st.sidebar.title("MNS News Analysis")
    st.sidebar.markdown("---")

    if df.empty:
        st.error(f"No data found. Path: {os.path.abspath(DATA_PATH)} (CWD: {os.getcwd()})")
        st.stop()

    # --- Global Search ---
    search_query = st.sidebar.text_input("🔍 Global Search", placeholder="Headline, reasoning, ticker...")
    if search_query:
        # Search across multiple columns
        mask = df.apply(lambda row: search_query.lower() in str(row['headline']).lower() or     
                                    search_query.lower() in str(row['description']).lower() or
                                    search_query.lower() in str(row['key_takeaways']).lower() or
                                    search_query.lower() in str(row['breaking_recap_reasoning']).lower() or
                                    search_query.lower() in str(row['news_worthy_reasoning']).lower() or
                                    search_query.lower() in str(row['actionable_intelligence']).lower() or
                                    search_query.lower() in str(row['us_ticker_name']).lower(), axis=1)
        df = df[mask]

    # --- Favorites Filter ---
    show_favorites = st.sidebar.checkbox("⭐ Show Favorites Only", value=False)
    if show_favorites:
        # Filter by IDs in favorites list
        # We need to ensure we are filtering based on the original index
        fav_ids = manager.favorites
        df = df[df.index.isin(fav_ids)]

    st.sidebar.markdown("---")

    # Ticker Filter
    tickers = ["All Tickers"] + sorted(df['us_ticker_name'].unique().tolist())
    selected_ticker = st.sidebar.selectbox("Select Ticker", tickers, index=0)
    
    # Filter DF by Ticker
    if selected_ticker == "All Tickers":
        ticker_df = df
    else:
        ticker_df = df[df['us_ticker_name'] == selected_ticker]
    
    # Date Filter
    # Get unique dates from the filtered dataframe
    dates = sorted(ticker_df['parsed_date'].dropna().unique(), reverse=True)
    view_df = ticker_df

    if dates:
        min_date = dates[-1]
        max_date = dates[0]
        
        date_mode = st.sidebar.selectbox(
            "Date Filter Mode", 
            ["Last 5 Days", "All History", "Specific Date", "Date Range"],
            index=0
        )
        
        if date_mode == "All History":
            view_df = ticker_df
        elif date_mode == "Last 5 Days":
            cutoff = max_date - timedelta(days=5)
            view_df = ticker_df[ticker_df['parsed_date'] >= cutoff]
        elif date_mode == "Specific Date":
            sel_date = st.sidebar.date_input(
                "Select Date", 
                value=max_date, 
                min_value=min_date, 
                max_value=max_date
            )
            view_df = ticker_df[ticker_df['parsed_date'] == sel_date]
        elif date_mode == "Date Range":
            # Default to last 7 days for the range view
            default_start = max_date - timedelta(days=7)
            sel_range = st.sidebar.date_input(
                "Select Range",
                value=(default_start, max_date),
                min_value=min_date,
                max_value=max_date
            )
            # st.date_input returns a tuple. It might be length 1 if user is still selecting.
            if isinstance(sel_range, tuple) and len(sel_range) == 2:
                start_d, end_d = sel_range
                view_df = ticker_df[(ticker_df['parsed_date'] >= start_d) & (ticker_df['parsed_date'] <= end_d)]
            elif isinstance(sel_range, tuple) and len(sel_range) == 1:
                 # Handle case where user picked start date but not end date yet
                 start_d = sel_range[0]
                 view_df = ticker_df[ticker_df['parsed_date'] >= start_d]

    st.sidebar.markdown("---")

    # Sentiment Filter
    # Create categorical column for filtering
    def categorize_sentiment(score):
        try:
            s = float(score)
            if s > 0.25: return "Positive"
            if s < -0.25: return "Negative"
            return "Neutral"
        except:
            return "Neutral"

    view_df['sentiment_category'] = view_df['news_sentiment'].apply(categorize_sentiment)
    
    all_sentiments = ["Positive", "Negative", "Neutral"]
    selected_sentiments = st.sidebar.multiselect("Filter by Sentiment", all_sentiments, default=[])
    if selected_sentiments:
        view_df = view_df[view_df['sentiment_category'].isin(selected_sentiments)]

    # Alignment Filter
    if not view_df.empty:
        # Create a copy to avoid SettingWithCopyWarning when adding new column
        view_df = view_df.copy()
        
        def get_alignment_status(row):
            md = manager.get_market_data(
                row['us_ticker_name'], 
                row['parsed_date'],
                local_ticker=row.get('local_ticker'),
                index_name=row.get('index'),
                sentiment_score=row.get('news_sentiment'),
                news_time_str=row.get('timestamp')
            )
            return md.get('sentiment_alignment', 'Unknown') if md else 'Unknown'

        view_df['alignment'] = view_df.apply(get_alignment_status, axis=1)
        
        all_alignments = ["Aligned", "Diverged", "Unknown"]
        selected_alignments = st.sidebar.multiselect("Filter by Alignment", all_alignments, default=[])
        if selected_alignments:
            view_df = view_df[view_df['alignment'].isin(selected_alignments)]


    # Source Filter
    all_sources = sorted(df['source_en'].dropna().unique().tolist())
    selected_sources = st.sidebar.multiselect("Select Source", all_sources, default=[])
    
    if selected_sources:
        view_df = view_df[view_df['source_en'].isin(selected_sources)]

    st.sidebar.markdown("---")

    # Breaking/Recap Filter
    show_only_breaking = st.sidebar.checkbox("Show Only Breaking Events", value=False)
    if show_only_breaking:
        view_df = view_df[view_df['breaking_recap'].str.upper() == 'BREAKING']

    # Significance Filter
    show_only_significant = st.sidebar.checkbox("Show Only Significant Events", value=True)
    if show_only_significant:
        view_df = view_df[view_df['classification'].str.upper() == 'SIGNIFICANT']

    # Duplicate Filter
    hide_duplicates = st.sidebar.checkbox("Hide User-Flagged Duplicates", value=True)
    # TODO: Implement actual filtering based on feedback_log.csv
    
    return view_df, selected_ticker

def render_metrics(view_df, manager):
    """Renders the top-level summary metrics."""
    # Calculate Bullish/Bearish counts dynamically using pre-calculated alignment
    bullish_count = 0
    bearish_count = 0
    
    if 'alignment' in view_df.columns:
        for _, row in view_df.iterrows():
            sentiment = row.get('sentiment_category', '')
            alignment = row['alignment']
            
            if sentiment == 'Positive' and alignment == 'Aligned':
                bullish_count += 1
            elif sentiment == 'Negative' and alignment == 'Aligned':
                bearish_count += 1

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Aligned Events", len(view_df))
    with col2:
        st.metric("Bullish Aligned Events", bullish_count)
    with col3:
        st.metric("Bearish Aligned Events", bearish_count)
    st.markdown("---")

# --- Main Execution ---
def main():
    load_css()
    manager = get_manager()
    df = manager.load_data()
    
    view_df, selected_ticker = render_sidebar(df, manager)
    
    st.title(f"News Analysis: {selected_ticker}")
    render_metrics(view_df, manager)
    
    for index, row in view_df.iterrows():
        render_ui.render_news_card(index, row, manager)

if __name__ == "__main__":
    main()
