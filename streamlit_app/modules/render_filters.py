import streamlit as st
import os
from datetime import datetime, timedelta
import re
import shlex
import unicodedata
import pandas as pd

def render_sidebar(df, manager, data_path):
    """Renders the sidebar filters and returns the filtered dataframe and selected ticker."""

    if df.empty:
        st.error(f"No data found. Path: {os.path.abspath(data_path)} (CWD: {os.getcwd()})")
        st.stop()
    
    st.sidebar.title("MNS News Analysis")    
    
    # Calculate GLOBAL date bounds once, so they don't change with filters
    all_dataset_dates = sorted(df['parsed_date'].dropna().unique())
    global_min_date = all_dataset_dates[0] if all_dataset_dates else None
    global_max_date = all_dataset_dates[-1] if all_dataset_dates else None
    
    st.sidebar.markdown("---")

    # --- Global Search (Placeholder for Filter later) ---
    search_query = st.sidebar.text_input("🔍 Global Search", placeholder="Headline, description, key_takeaways, reasoning, ticker...")

    # --- Favorites Filter ---
    show_favorites = st.sidebar.checkbox("⭐ Show Favorites Only", value=False)
    if show_favorites:
        # Filter by IDs in favorites list
        # We need to ensure we are filtering based on the original index
        fav_ids = manager.favorites
        df = df[df.index.isin(fav_ids)]

    st.sidebar.markdown("---")

    # Ticker Filter
    tickers = ["All Tickers"] + sorted(df['us_ticker_name'].dropna().astype(str).unique().tolist())
    selected_ticker = st.sidebar.selectbox("Select Ticker", tickers, index=0)
    
    # Filter DF by Ticker (the downstream df will be ticker_df)
    if selected_ticker == "All Tickers":
        ticker_df = df
    else:
        ticker_df = df[df['us_ticker_name'] == selected_ticker]
    
    # Date Filter
    # Use global dates for stability
    view_df = ticker_df

    if global_max_date:
        min_date = global_min_date
        max_date = global_max_date
        
        # Default date filter will be "Last 5 Days"
        last_5_cutoff = max_date - timedelta(days=5)
        date_mode = st.sidebar.selectbox(
            "Date Filter Mode", 
            [
                f"Last 5 Days ({last_5_cutoff.strftime('%b %d')} - {max_date.strftime('%b %d')})", 
                "All History", 
                "Specific Date", 
                "Date Range"
            ],
            index=0
        )
        
        # Normalize the choice back to a key
        mode_key = "Last 5 Days" if "Last 5 Days" in date_mode else date_mode

        if mode_key == "All History":
            view_df = ticker_df
        elif mode_key == "Last 5 Days":
            view_df = ticker_df[ticker_df['parsed_date'] >= last_5_cutoff]
        elif mode_key == "Specific Date":
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

    view_df = view_df.copy()
    view_df['sentiment_category'] = view_df['news_sentiment'].apply(categorize_sentiment)
    
    all_sentiments = ["Positive", "Negative", "Neutral"]
    selected_sentiments = st.sidebar.multiselect("Filter by Sentiment", all_sentiments, default=[])
    if selected_sentiments:
        view_df = view_df[view_df['sentiment_category'].isin(selected_sentiments)]

    # Alignment Filter
    if not view_df.empty:
        
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

        # Generate a new column for alignment based on market_data df
        view_df['alignment'] = view_df.apply(get_alignment_status, axis=1)
        
        all_alignments = ["Aligned", "Diverged"]
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

    # News Lifecycle Filters
    hide_linked_recaps = st.sidebar.checkbox("Hide Linked Recaps (Stacked)", value=True, help="Hide articles that have been manually linked to a parent story to declutter the timeline.")
    if hide_linked_recaps:
        # We check if the row HAS a valid linked_original_id
        # Note: linked_original_id is None-equivalent if not linked with other articles
        view_df = view_df[view_df['linked_original_id'].isna() | (view_df['linked_original_id'] == 'None')]

    # Duplicate Filter (May be redundant with linked recaps and deprecated)
    hide_duplicates = st.sidebar.checkbox("Hide User-Flagged Duplicates", value=True)

    st.sidebar.markdown("---")
    
    # --- Apply Global Search ---
    if search_query:
        try:
            tokens = shlex.split(search_query.lower())
        except ValueError:
            tokens = search_query.lower().split()
        tokens = [t.strip() for t in tokens if t.strip()]

        if tokens:
            def normalize_text(t):
                if not t: return ""
                t = unicodedata.normalize('NFKC', str(t))
                return " ".join(t.split()).lower()

            def get_row_search_data(row):
                field_map = {
                    'Headline': row.get('headline', ''),
                    'Description': row.get('description', ''),
                    'Key Takeaways': row.get('key_takeaways', ''),
                    'Model Reasoning': f"{row.get('breaking_recap_reasoning', '')} {row.get('news_worthy_reasoning', '')}",
                    'Intelligence': row.get('actionable_intelligence', ''),
                    'Ticker': row.get('us_ticker_name', ''),
                }
                norm_map = {k: normalize_text(v) for k, v in field_map.items()}
                full_text = " ".join(norm_map.values())
                
                for token in tokens:
                    if not re.search(rf"(?<!\w){re.escape(normalize_text(token))}(?!\w)", full_text):
                        return None
                
                matches = [label for label, text in norm_map.items() if any(re.search(rf"(?<!\w){re.escape(normalize_text(t))}(?!\w)", text) for t in tokens)]
                return matches

            # First, see how many matches exist in the TOTAL dataset for the 'Smart Hint'
            # We use 'df' which is unfiltered by date/ticker
            df['all_search_matches'] = df.apply(get_row_search_data, axis=1)
            total_matches_count = df['all_search_matches'].notna().sum()

            # Now, apply the search TO THE CURRENT VIEW (after date/ticker filters)
            view_df['search_matches'] = view_df.apply(get_row_search_data, axis=1)
            view_df = view_df[view_df['search_matches'].notna()]
            visible_count = len(view_df)

            # --- Smart Search Hint ---
            if total_matches_count > 0:
                hidden_count = total_matches_count - visible_count
                if hidden_count > 0:
                    hint_msg = f"💡 {total_matches_count} matches found for '{search_query}'."
                    if visible_count > 0:
                        hint_msg += f" (Showing {visible_count}, {hidden_count} are hidden by current filters)"
                    else:
                        hint_msg += f" All {total_matches_count} are currently hidden by Date/Ticker filters."
                    st.sidebar.warning(f"{hint_msg} Switch to 'All History' or 'All Tickers' to see them.")

    # --- FINAL CHRONOLOGICAL SORT ---
    # Ensure any filtering didn't shuffle the order
    if not view_df.empty:
        # Create a temporary datetime for sorting
        def get_sort_dt_global(r):
            d = r.get('parsed_date')
            if pd.isna(d): return pd.Timestamp.min
            ts = r.get('timestamp')
            if pd.notna(ts) and str(ts).strip():
                try:
                    t_obj = pd.to_datetime(str(ts), errors='coerce')
                    if pd.notna(t_obj): return datetime.combine(d, t_obj.time())
                except: pass
            return datetime.combine(d, datetime.min.time())
            
        view_df = view_df.copy()
        view_df['_global_sort_dt'] = view_df.apply(get_sort_dt_global, axis=1)
        view_df = view_df.sort_values('_global_sort_dt', ascending=False).drop(columns=['_global_sort_dt'])

    # --- FINAL REFRESH BUTTON ---
    _ , col2, _ = st.sidebar.columns([0.5, 2, 0.5])
    with col2:
        if st.button("🔄 Refresh Data", help="Clear cache and reload data from BigQuery/CSV", width='stretch'):
            st.cache_data.clear()
            st.rerun()

    return view_df, selected_ticker

