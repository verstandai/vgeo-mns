import streamlit as st
# Force Reload
import pandas as pd
import os
from datetime import datetime, timedelta
from data_manager import DataManager

# --- Configuration ---
PAGE_TITLE = "MNS | Sentiment Validation"
PAGE_ICON = "📈"
DATA_PATH = "../mns_demo_output.csv"
FEEDBACK_PATH = "feedback_log.csv"

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
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

@st.cache_resource
def get_manager():
    """Initializes and caches the DataManager."""
    # Construct absolute paths
    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, "..", "mns_demo_output.csv")
    feedback_path = os.path.join(base_dir, "feedback_log.csv")
    
    return DataManager(data_path, feedback_path)

def get_metric_color(val):
    """Returns CSS class for metric coloring."""
    return "metric-positive" if val > 0 else "metric-negative"

def render_sidebar(df, manager):
    """Renders the sidebar filters and returns the filtered dataframe and selected ticker."""
    st.sidebar.title("MNS News Analysis")
    st.sidebar.markdown("---")

    if df.empty:
        st.error("No data found. Please ensure mns_demo_output.csv is in the parent directory.")
        st.stop()

    # --- Global Search ---
    search_query = st.sidebar.text_input("🔍 Global Search", placeholder="Headline, reasoning, ticker...")
    if search_query:
        # Search across multiple columns
        mask = df.apply(lambda row: search_query.lower() in str(row['headline']).lower() or 
                                    search_query.lower() in str(row['reasoning']).lower() or
                                    search_query.lower() in str(row['key_factors']).lower() or
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
        date_options = ["Last 5 Days", "All History"] + [d.strftime('%Y-%m-%d') for d in dates]
        selected_date_option = st.sidebar.selectbox("Select Date Range", date_options, index=0)
        
        if selected_date_option == "All History":
            view_df = ticker_df
        elif selected_date_option == "Last 5 Days":
            max_date = dates[0]
            cutoff = max_date - timedelta(days=5)
            view_df = ticker_df[ticker_df['parsed_date'] >= cutoff]
        else:
            sel_date = datetime.strptime(selected_date_option, '%Y-%m-%d').date()
            view_df = ticker_df[ticker_df['parsed_date'] == sel_date]

    st.sidebar.markdown("---")

    # Sentiment Filter
    all_sentiments = sorted(df['news_sentiment'].unique().tolist())
    selected_sentiments = st.sidebar.multiselect("Filter by Sentiment", all_sentiments, default=[])
    if selected_sentiments:
        view_df = view_df[view_df['news_sentiment'].isin(selected_sentiments)]

    # Event Correlation Filter (Dropdown)
    # Get unique non-null values
    all_correlations = sorted(df['event_correlation'].dropna().unique().tolist())
    selected_correlations = st.sidebar.multiselect("Filter by Event Correlation", all_correlations, default=[])
    if selected_correlations:
        view_df = view_df[view_df['event_correlation'].isin(selected_correlations)]

    st.sidebar.markdown("---")

    # Significance Filter
    show_only_significant = st.sidebar.checkbox("Show Only Significant Events", value=False)
    if show_only_significant:
        view_df = view_df[view_df['classification'] == 'SIGNIFICANT']

    # Duplicate Filter
    hide_duplicates = st.sidebar.checkbox("Hide User-Flagged Duplicates", value=True)
    # TODO: Implement actual filtering based on feedback_log.csv
    
    return view_df, selected_ticker

def render_metrics(view_df):
    """Renders the top-level summary metrics."""
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Events", len(view_df))
    with col2:
        bullish = len(view_df[view_df['news_sentiment'] == 'POSITIVE'])
        st.metric("Bullish Signals", bullish)
    with col3:
        bearish = len(view_df[view_df['news_sentiment'] == 'NEGATIVE'])
        st.metric("Bearish Signals", bearish)
    st.markdown("---")

def render_news_card(index, row, manager):
    """Renders a single news card with details and feedback form."""
    # Pass local ticker and index name for accurate data fetching
    market_data = manager.get_market_data(
        row['us_ticker_name'], 
        row['parsed_date'], 
        local_ticker=row.get('local_ticker_name'), 
        index_name=row.get('index')
    )
    
    sentiment_color = "badge-positive" if row['news_sentiment'] == 'POSITIVE' else "badge-negative" if row['news_sentiment'] == 'NEGATIVE' else "badge-insignificant"
    sig_badge = "badge-significant" if row['classification'] == 'SIGNIFICANT' else "badge-insignificant"
    
    # Format Date
    display_date = row['parsed_date'].strftime('%b %d, %Y').upper() if row['parsed_date'] else row['date']

    # Favorite Logic
    is_fav = manager.is_favorite(index)
    fav_icon = "⭐" if is_fav else "☆"
    fav_label = "Saved" if is_fav else "Save"
    
    with st.container():
        # Header Row with Favorite Button
        c_head, c_fav = st.columns([8, 1])
        with c_head:
            st.markdown(f"""
            <div class="news-card">
                <div class="card-header">
                    <div class="header-left">
                        <span class="news-date">{display_date}</span>
                        <span class="ticker-badge">{row['us_ticker_name']}</span>
                        <span class="company-name">{row.get('company_name', '')}</span>
                    </div>
                    <div>
                        <span class="badge {sentiment_color}">{row['news_sentiment']}</span>
                        <span class="badge {sig_badge}">{row['classification']}</span>
                    </div>
                </div>
                <div class="news-headline">{row['headline']}</div>
                <div class="news-source">Source: {row['source']} | Reporter: {row.get('reporter', 'Unknown')}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with c_fav:
            # Use a unique key for each button
            if st.button(f"{fav_icon}", key=f"fav_btn_{index}", help="Toggle Favorite"):
                manager.toggle_favorite(index)
                st.rerun()
        
        c1, c2 = st.columns([3, 2])
        
        with c1:
            st.markdown("#### Analysis")
            st.markdown(f"**Reasoning:** {row['reasoning']}")
            st.markdown(f"**Key Factors:** {row['key_factors']}")
            
        with c2:
            st.markdown("#### Market Impact (Day of Event)")
            if market_data:
                # Row 1: Stock & Index Change
                m1, m2 = st.columns(2)
                m1.markdown(f"<div class='metric-label'>Stock Chg</div><div class='metric-value {get_metric_color(market_data.get('pct_change_day', 0))}'>{market_data.get('pct_change_day', 0):.2f}%</div>", unsafe_allow_html=True)
                m2.markdown(f"<div class='metric-label'>Index Chg</div><div class='metric-value {get_metric_color(market_data.get('index_pct_change', 0))}'>{market_data.get('index_pct_change', 0):.2f}%</div>", unsafe_allow_html=True)
                
                # Row 2: Relative & Volume
                m3, m4 = st.columns(2)
                m3.markdown(f"<div class='metric-label'>Rel Change</div><div class='metric-value {get_metric_color(market_data.get('relative_change', 0))}'>{market_data.get('relative_change', 0):.2f}%</div>", unsafe_allow_html=True)
                m4.markdown(f"<div class='metric-label'>Vol Ratio</div><div class='metric-value'>{market_data.get('volume_rel', 0):.2f}x</div>", unsafe_allow_html=True)
                
                if market_data.get('is_mock'):
                    st.caption("⚠️ Market data simulated")
            else:
                st.info("Market data unavailable for this date.")

        with st.expander("Model Validation", expanded=False):
            # Dynamic Feedback (No Form to allow interactivity)
            # Use columns for better layout
            f1, f2 = st.columns(2)
            
            with f1:
                # Sentiment Validation
                st.caption(f"Model Sentiment: **{row['news_sentiment']}**")
                sent_correct = st.radio("Is Sentiment Correct?", ["Yes", "No"], horizontal=True, key=f"sent_check_{index}")
                
                final_sentiment = row['news_sentiment']
                if sent_correct == "No":
                    final_sentiment = st.selectbox("Select Correct Sentiment", ["POSITIVE", "NEUTRAL", "NEGATIVE"], key=f"sent_fix_{index}")
            
            with f2:
                # Event Correlation (Moved Up)
                st.caption("Event Correlation")
                corr_correct = st.radio("Is Event Correlation Correct?", ["Yes", "No"], horizontal=True, key=f"corr_check_{index}")
                
                final_corr = "Correct"
                if corr_correct == "No":
                    final_corr = st.selectbox("Select Correct Correlation", ["Strong", "Medium", "Weak"], key=f"corr_fix_{index}")

            # Row 2: Source & Significance Validation
            f3, f4 = st.columns(2)
            
            with f3:
                # Source Utility
                st.caption("Source Quality")
                source_useful = st.radio("Is Source/Reporter Useful?", ["Yes", "No"], horizontal=True, key=f"src_check_{index}")

            with f4:
                # Significance Validation (Moved Down)
                st.caption(f"Model Significance: **{row['classification']}**")
                sig_correct = st.radio("Is Significance Correct?", ["Yes", "No"], horizontal=True, key=f"sig_check_{index}")
                
                final_sig = row['classification']
                if sig_correct == "No":
                    # Mapping HIGH/LOW to internal values if needed, or just using them directly
                    # Assuming HIGH = SIGNIFICANT, LOW = INSIGNIFICANT for consistency with data
                    sig_option = st.selectbox("Select Correct Significance", ["HIGH", "LOW"], key=f"sig_fix_{index}")
                    final_sig = "SIGNIFICANT" if sig_option == "HIGH" else "INSIGNIFICANT"

            st.caption("Additional Feedback")
            f_dup = st.checkbox("Mark as Duplicate", key=f"dup_{index}")
            f_notes = st.text_area("Notes", placeholder="Reasoning for correction...", height=80, key=f"note_{index}")
            
            if st.button("Submit Feedback", key=f"btn_{index}"):
                feedback_entry = {
                    "news_id": index,
                    "ticker": row['us_ticker_name'],
                    "date": row['date'],
                    "headline": row['headline'],
                    "user_sentiment_correction": final_sentiment if sent_correct == "No" else "Correct",
                    "user_significance_correction": final_sig if sig_correct == "No" else "Correct",
                    "source_useful": source_useful,
                    "event_correlation_correction": final_corr,
                    "is_duplicate": f_dup,
                    "notes": f_notes
                }
                manager.save_feedback(feedback_entry)
                st.success("Saved!")

        with st.expander("Show Source Details"):
            st.markdown(f"**Original Headline:** {row['original_headline']}")
            st.markdown(f"**Link:** {row['url']}")
    st.markdown("---")

# --- Main Execution ---
def main():
    load_css()
    manager = get_manager()
    df = manager.load_data()
    
    view_df, selected_ticker = render_sidebar(df, manager)
    
    st.title(f"News Analysis: {selected_ticker}")
    render_metrics(view_df)
    
    for index, row in view_df.iterrows():
        render_news_card(index, row, manager)

if __name__ == "__main__":
    main()
