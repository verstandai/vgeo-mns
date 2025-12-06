import streamlit as st
# Force Reload
import pandas as pd
import os
import html
import altair as alt
from datetime import datetime, timedelta
from data_manager import DataManager

# --- Configuration ---
PAGE_TITLE = "MNS | Sentiment Validation"
PAGE_ICON = "📈"
DATA_PATH = "../mns_demo_enriched.csv"
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
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True) #allow injection of customized html 

@st.cache_resource
def get_manager():
    """Initializes and caches the DataManager."""
    # Construct absolute paths
    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, "..", "mns_demo_enriched.csv")
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
        st.error("No data found. Please ensure mns_demo_enriched.csv is in the parent directory.")
        st.stop()

    # --- Global Search ---
    search_query = st.sidebar.text_input("🔍 Global Search", placeholder="Headline, reasoning, ticker...")
    if search_query:
        # Search across multiple columns
        mask = df.apply(lambda row: search_query.lower() in str(row['headline']).lower() or     
                                    search_query.lower() in str(row['description']).lower() or
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
        min_date = dates[-1]
        max_date = dates[0]
        
        date_mode = st.sidebar.selectbox(
            "Date Filter Mode", 
            ["All History", "Last 5 Days", "Specific Date", "Date Range"],
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

    # Event Correlation Filter (Dropdown)
    # Get unique non-null values
    all_correlations = sorted(df['event_correlation'].dropna().unique().tolist())
    selected_correlations = st.sidebar.multiselect("Filter by Event Correlation", all_correlations, default=[])
    if selected_correlations:
        view_df = view_df[view_df['event_correlation'].isin(selected_correlations)]

    # Alignment Filter
    if not view_df.empty:
        # Create a copy to avoid SettingWithCopyWarning when adding new column
        view_df = view_df.copy()
        
        def get_alignment_status(row):
            md = manager.get_market_data(
                row['us_ticker_name'], 
                row['parsed_date'],
                local_ticker=row.get('local_ticker_name'),
                index_name=row.get('index'),
                sentiment_score=row.get('news_sentiment')
            )
            return md.get('sentiment_alignment', 'Unknown') if md else 'Unknown'

        view_df['alignment'] = view_df.apply(get_alignment_status, axis=1)
        
        all_alignments = ["Aligned", "Divergent", "Unknown"]
        selected_alignments = st.sidebar.multiselect("Filter by Alignment", all_alignments, default=[])
        if selected_alignments:
            view_df = view_df[view_df['alignment'].isin(selected_alignments)]

    st.sidebar.markdown("---")

    # Breaking/Recap Filter
    show_only_breaking = st.sidebar.checkbox("Show Only Breaking Events", value=False)
    if show_only_breaking:
        view_df = view_df[view_df['breaking_recap'] == 'Breaking']

    # Significance Filter
    show_only_significant = st.sidebar.checkbox("Show Only Significant Events", value=True)
    if show_only_significant:
        view_df = view_df[view_df['classification'] == 'SIGNIFICANT']

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
        st.metric("Total Events", len(view_df))
    with col2:
        st.metric("Bullish Signals", bullish_count)
    with col3:
        st.metric("Bearish Signals", bearish_count)
    st.markdown("---")

def render_news_card(index, row, manager):
    """Renders a single news card with details and feedback form."""
    # Pass local ticker and index name    # Fetch Market Data (if available)
    market_data = manager.get_market_data(
        row['us_ticker_name'], 
        row['parsed_date'],
        local_ticker=row.get('local_ticker_name'),
        index_name=row.get('index'),
        sentiment_score=row.get('news_sentiment')
    )
    
    # Determine Sentiment Badge
    sent_cat = row.get('sentiment_category', 'Neutral')
    if sent_cat == 'Positive':
        sentiment_color = "badge-positive"
    elif sent_cat == 'Negative':
        sentiment_color = "badge-negative"
    else:
        sentiment_color = "badge-insignificant"

    # Use calculated impact strength if available, otherwise fallback to LLM's correlation
    if market_data and 'impact_strength' in market_data:
        corr_cat = market_data['impact_strength']
    else:
        corr_cat = row.get('event_correlation', 'N/A')

    if corr_cat == 'Strong':
        corr_color = "badge-strong"
    elif corr_cat == 'Medium':
        corr_color = "badge-medium"
    elif corr_cat == 'Weak':
        corr_color = "badge-weak"
    else:
        corr_color = "badge-insignificant"

    sig_badge = "badge-significant" if row['classification'] == 'SIGNIFICANT' else "badge-insignificant"
    recap_badge = "badge-breaking" if row['breaking_recap'] == 'Breaking' else "badge-insignificant" if row['breaking_recap'] == 'Recap' else "badge-insignificant"

    # Format Date
    display_date = row['parsed_date'].strftime('%B %d, %Y') if row['parsed_date'] else row['date']

    # Favorite Logic
    is_fav = manager.is_favorite(index)
    fav_icon = "⭐" if is_fav else "☆"
    fav_label = "Saved" if is_fav else "Save"
    
    # Escape HTML content to prevent rendering issues (e.g., <From Minkabu>)
    safe_headline = html.escape(str(row['headline']))
    safe_company = html.escape(str(row.get('company_name', '')))
    safe_source = html.escape(str(row['source']))
    safe_reporter = html.escape(str(row.get('reporter', 'Unknown')))

    with st.container():
        # Header Row with Favorite Button
        c_head, c_fav = st.columns([8, 1])
        with c_head:
            # Determine badges
            category_badge = f'<span class="badge badge-insignificant">{row.get("event_category", "Unknown")}</span>'
            recap_badge_html = f'<span class="badge {recap_badge}">{row.get("breaking_recap", "Unknown").upper()}</span>'
            sig_badge_html = f'<span class="badge {sig_badge}">{row.get("classification", "Unknown")}</span>'

            st.markdown(f"""
            <div class="news-card">
                <div class="card-header">
                    <div class="header-left">
                        <span class="news-date">{display_date}</span>
                        <span class="ticker-badge">{row['us_ticker_name']}</span>
                        <span class="company-name">{safe_company}</span>
                    </div>
                    <div>
                        {recap_badge_html}
                        {sig_badge_html}
                    </div>
                </div>
                <div class="news-headline">{safe_headline}</div>
                <div class="news-source">Source: {safe_source} | Reporter: {safe_reporter}</div>
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
            if pd.notna(row.get('description')):
                st.markdown(f"**Description:** {row['description']}")
            st.markdown(f"**Reasoning:** {row['reasoning']}")
            st.markdown(f"**Key Factors:** {row['key_factors']}")
            
        with c2:
                        
            st.markdown("#### Event Classification")
            e1, e2 = st.columns(2)
            with e1:
                st.markdown(f"<div class='metric-label'>Category</div><div class='metric-value-small'>{row.get('event_category', 'N/A')}</div>", unsafe_allow_html=True)
                st.markdown("")
                st.markdown(f"<div class='metric-label'>Sentiment</div><span class=\"badge {sentiment_color}\" style='margin-left: 0;'>{sent_cat.upper()} ({row['news_sentiment']:.2f})</span>", unsafe_allow_html=True)                


            with e2:
                st.markdown(f"<div class='metric-label'>Event to Market Impact</div><span class=\"badge {corr_color}\" style='margin-left: 0;'>{corr_cat.upper()}</span>", unsafe_allow_html=True)
                
                if market_data and 'sentiment_alignment' in market_data:
                    align_status = market_data['sentiment_alignment']
                    align_color = "#4cd964" if align_status == "Aligned" else "#ff3b30" if align_status == "Diverged" else "#a0a0a0"
                    st.markdown("")
                    st.markdown(f"<div class='metric-label'>Alignment</div><div class='metric-value-small' style='color: {align_color};'>{align_status}</div>", unsafe_allow_html=True)

            st.markdown('<div style="border-top: 1px solid #444; margin: 15px 0;"></div>', unsafe_allow_html=True)
            st.markdown("#### Market Impact (Day of Event)")
            if market_data:
                # Helper for delta color
                def delta_span(val):
                    color = "metric-positive" if val > 0 else "metric-negative" if val < 0 else ""
                    arrow = "↑" if val > 0 else "↓" if val < 0 else ""
                    return f"<span class='{color}' style='font-size: 0.85em; margin-left: 4px;'>{arrow} {abs(val):.2f}%</span>"

                # Row 1: Stock & Index Change
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.markdown(f"<div class='metric-label'>Stock Close</div><div class='metric-value-small'>{market_data['close']:.2f} {delta_span(market_data['pct_change_day'])}</div>", unsafe_allow_html=True)
                with m2:
                    st.markdown(f"<div class='metric-label'>Index Close</div><div class='metric-value-small'>{market_data['index_close']:.2f} {delta_span(market_data['index_pct_change'])}</div>", unsafe_allow_html=True)
                with m3:
                    st.markdown(f"<div class='metric-label'>Rel. to Index</div><div class='metric-value-small'>{market_data['relative_change']:.2f}%</div>", unsafe_allow_html=True)
                
                st.markdown("") # Spacer

                # Row 2: Intraday, Sigma, Vol
                m4, m5, m6 = st.columns(3)
                with m4:
                    stock_intra = market_data.get('intraday_change', 0)
                    rel_intra = market_data.get('relative_intraday', 0)
                    st.markdown(f"<div class='metric-label'>Intraday (Stock/Rel)</div><div class='metric-value-small'>{delta_span(stock_intra)} / {delta_span(rel_intra)}</div>", unsafe_allow_html=True)
                with m5:
                    sigma = market_data.get('sigma_move', 0)
                    sigma_color = "metric-positive" if sigma > 0 else "metric-negative"
                    st.markdown(f"<div class='metric-label'>Sigma Move (Last 30d)</div><div class='metric-value-small'><span class='{sigma_color}'>{sigma:+.1f}σ</span></div>", unsafe_allow_html=True)
                with m6:
                    st.markdown(f"<div class='metric-label'>Vol Ratio (Last 20d)</div><div class='metric-value-small'>{market_data['volume_rel']:.1f}x</div>", unsafe_allow_html=True)

                st.markdown("") # Spacer

                # Row 3: CAR
                m7, m8, m9 = st.columns(3)
                with m7:
                    if 'car_pre_3d' in market_data:
                        st.markdown(f"<div class='metric-label'>Pre-Event CAR (T-3)</div><div class='metric-value-small'>{market_data['car_pre_3d']:.2f}%</div>", unsafe_allow_html=True)
                with m8:
                    if 'car_3d' in market_data:
                        st.markdown(f"<div class='metric-label'>Post-Event CAR (T+3)</div><div class='metric-value-small'>{market_data['car_3d']:.2f}%</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div class='metric-label'>Abnormal Ret</div><div class='metric-value-small'>{market_data.get('abnormal_return', 0):.2f}%</div>", unsafe_allow_html=True)
                with m9:
                    st.markdown(f"<div class='metric-label'>Gap %</div><div class='metric-value-small'>{delta_span(market_data.get('gap_pct', 0))}</div>", unsafe_allow_html=True)
                
                st.markdown("")
                if market_data.get('is_mock'):
                    st.caption("⚠️ Market data simulated")
            else:
                st.info("Market data unavailable for this date.")
            st.markdown("") 

        # Price Chart Expander
        if market_data and 'chart_data' in market_data and market_data['chart_data'] is not None:
            with st.expander("Price Trend (T-40 to T+10)", expanded=False):
                # Prepare data for Altair
                chart_df = market_data['chart_data'].reset_index()
                # Rename first column to Date (it might be 'index' or 'Date')
                chart_df.rename(columns={chart_df.columns[0]: 'Date'}, inplace=True)
                # Ensure Date is datetime for Altair and comparison
                chart_df['Date'] = pd.to_datetime(chart_df['Date'])
                
                # Melt for Altair (combining different assets into a single column)
                chart_df = chart_df.melt('Date', var_name='Asset', value_name='Return')
                
                # Create Chart
                base = alt.Chart(chart_df).encode(
                    x=alt.X('Date:T', axis=alt.Axis(format='%m/%d/%y', title='Date'))
                )

                lines = base.mark_line(point=True).encode(
                    y=alt.Y('Return:Q', axis=alt.Axis(format='.0f'), title='Return (%)'),
                    color=alt.Color('Asset:N', legend=alt.Legend(title=None, orient='top')),
                    tooltip=[
                        alt.Tooltip('Date:T', format='%m/%d/%y', title='Date'),
                        alt.Tooltip('Asset:N', title='Asset'),
                        alt.Tooltip('Return:Q', format='.2f', title='Return (%)')
                    ]
                )
                
                # Define vertical rules for T-3 and T+3
                rules_data = []
                if market_data.get('date_t_minus_3'):
                    rules_data.append({'Date': pd.Timestamp(market_data['date_t_minus_3']), 'Color': 'red', 'Label': 'T-3'})
                if market_data.get('date_t_plus_3'):
                    rules_data.append({'Date': pd.Timestamp(market_data['date_t_plus_3']), 'Color': 'red', 'Label': 'T+3'})
                rules_df = pd.DataFrame(rules_data)
                rules = alt.Chart(rules_df).mark_rule(strokeDash=[4, 4]).encode(
                    x='Date:T',
                    color=alt.Color('Color:N', scale=None),
                    tooltip=['Label:N']
                )
                # Rule Labels
                rule_labels = alt.Chart(rules_df).mark_text(
                    align='left',
                    baseline='top',
                    dx=5,
                    dy=5,
                    fontSize=11
                ).encode(
                    x='Date:T',
                    y=alt.value(0),
                    text='Label:N',
                    color=alt.Color('Color:N', scale=None)
                )
                
                # Highlight Event Day Points
                event_df = chart_df[chart_df['Date'] == pd.Timestamp(market_data.get('date_event'))]
                points = alt.Chart(event_df).mark_point(filled=True, size=100, color='yellow', stroke='black', strokeWidth=2).encode(
                    x='Date:T',
                    y='Return:Q',
                    tooltip=[
                        alt.Tooltip('Date:T', format='%m/%d/%y', title='Date'),
                        alt.Tooltip('Asset:N', title='Asset'),
                        alt.Tooltip('Return:Q', format='.2f', title='Return (%)')
                    ]
                )
                
                # Event Day Label
                event_label = alt.Chart(event_df).mark_text(
                    align='center',
                    baseline='bottom',
                    dy=-10, # Shift up
                    fontSize=11,
                    fontWeight='bold',
                    color='white'
                ).encode(
                    x='Date:T',
                    y='Return:Q',
                    text=alt.value('Event Day')
                )

                chart = alt.layer(lines, rules, rule_labels, points, event_label).properties(height=400).interactive()
                
                st.altair_chart(chart, use_container_width=True)

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
            f_recap = st.checkbox("Mark as Recap", key=f"recap_{index}")
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
                    "is_recap": f_recap,
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
    render_metrics(view_df, manager)
    
    for index, row in view_df.iterrows():
        render_news_card(index, row, manager)

if __name__ == "__main__":
    main()
