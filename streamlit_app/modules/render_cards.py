import streamlit as st
import pandas as pd
import html
import textwrap
import os
from datetime import datetime
import pytz
from modules import charts

def delta_span(val):
    if val is None: return ""
    color = "metric-positive" if val > 0 else "metric-negative" if val < 0 else ""
    arrow = "↑" if val > 0 else "↓" if val < 0 else ""
    return f"<span class='{color}' style='font-size: 0.85em; margin-left: 4px;'>{arrow} {abs(val):.2f}%</span>"

def fmt_vol(v):
    if v >= 1_000_000: return f"{v/1_000_000:.1f}M"
    elif v >= 1_000: return f"{v/1_000:.1f}K"
    return f"{v:.0f}"

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
        st.metric("Bullish Aligned Events", bullish_count)
    with col3:
        st.metric("Bearish Aligned Events", bearish_count)
    st.markdown("---")

def render_news_card(index, row, manager, children_recaps=None):
    """Renders a single news card with details, feedback form, and lifecycle grid."""
    if children_recaps is None: children_recaps = []
    
    # Fetch Market Data
    market_data = manager.get_market_data(
        row['us_ticker_name'], 
        row['parsed_date'],
        local_ticker=row.get('local_ticker'),
        index_name=row.get('index'),
        sentiment_score=row.get('news_sentiment'),
        news_time_str=row.get('timestamp')
    )
    
    # Determine Sentiment Badge
    sent_cat = row.get('sentiment_category', 'Neutral')
    sentiment_color = "badge-positive" if sent_cat == 'Positive' else "badge-negative" if sent_cat == 'Negative' else "badge-insignificant"

    # Badges - Notice we use row['breaking_recap'] which has been overridden in app.py
    sig_badge = "badge-significant" if row['classification'].upper() == 'SIGNIFICANT' else "badge-insignificant"
    recap_status = row.get('breaking_recap', 'Unknown').upper()
    recap_badge = "badge-breaking" if recap_status == 'BREAKING' else "badge-insignificant"

    # Format Date & Time
    date_str = row['parsed_date'].strftime('%B %d, %Y') if row['parsed_date'] else row['date']
    
    time_str = ""
    ts_val = row.get('timestamp')
    if pd.notna(ts_val) and str(ts_val).strip() != '':
        try:
            # Parse timestamp (handles both "14:30" CSV format and full datetime from BigQuery)
            ts_obj = pd.to_datetime(str(ts_val), errors='coerce')
            if pd.notna(ts_obj):
                eastern = pytz.timezone('America/New_York')
                jst = pytz.timezone('Asia/Tokyo')
                article_date = pd.to_datetime(row.get('parsed_date')).date()
                naive_dt = datetime.combine(article_date, ts_obj.time())
                est_dt = eastern.localize(naive_dt)
                jst_dt = est_dt.astimezone(jst)
                est_str = est_dt.strftime('%I:%M %p')
                jst_str = jst_dt.strftime('%I:%M %p')
                day_diff = (jst_dt.date() - est_dt.date()).days
                if day_diff > 0: jst_str += " (+1)"
                elif day_diff < 0: jst_str += " (-1)"
                time_str = f" | {est_str} EST / {jst_str} JST"
            else:
                time_str = f" | {ts_val}"
        except:
            time_str = f" | {ts_val}"
    display_date = f"{date_str}{time_str}"

    # Favorite Logic
    is_fav = manager.is_favorite(index)
    fav_icon = "⭐" if is_fav else "☆"
    
    # To prevent broken formatting as the code is directly injecting headlines into html
    safe_headline = html.escape(str(row['headline']))
    safe_company = html.escape(str(row.get('company_name', '')))
    safe_source = html.escape(str(row['source_en']))
    safe_reporter = html.escape(str(row.get('reporter', 'Unknown')))

    with st.container():
        # News Header Row
        # Card (Left, Large) | Buttons (Right, Small)
        c_head, c_archive, c_fav = st.columns([20, 1, 1])
        
        with c_head:
            # Recap Link Logic (For the Recap Story)
            link_html = ""
            linked_id = row.get('linked_original_id')
            if pd.notna(linked_id) and linked_id != "None" and not manager.full_df.empty:
                try:
                    orig_row = manager.full_df.loc[int(float(linked_id))]
                    orig_headline = html.escape(orig_row['headline'])
                    link_html = f'<div style="font-size: 0.8rem; color: #4CAF50; margin-bottom: 8px; font-style: italic;">🔗 Recap of: <b>{orig_headline}</b></div>'
                except:
                    pass

            # Children Counter Badge (For the Original Story)
            children_badge_html = ""
            if children_recaps:
                children_badge_html = f'<span class="badge badge-insignificant" style="background:#444; color:#fff; border:1px solid #666;">🔄 {len(children_recaps)} RECAPS</span>'

            recap_badge_html = f'<span class="badge {recap_badge}">{recap_status}</span>'
            sig_badge_html = f'<span class="badge {sig_badge}">{row.get("classification", "Unknown").upper()}</span>'

            st.markdown(textwrap.dedent(f"""
                <div class="news-card">
                <div class="card-header" style="margin-bottom: 4px;">
                <div class="header-left">
                <span class="news-date">{display_date}</span>
                </div>
                <div style="display:flex; gap:5px;">
                {children_badge_html}
                {recap_badge_html}
                {sig_badge_html}
                </div>
                </div>
                {link_html}
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
                <span class="ticker-badge" style="margin-left: 0;">{row['us_ticker_name']}</span>
                <span class="company-name">{safe_company}</span>
                </div>
                <div class="news-headline">{safe_headline}</div>
                <div class="news-source">Source: {safe_source} | Reporter: {safe_reporter}</div>
                </div>
                """), unsafe_allow_html=True)
            
            # --- The "Stack" of Recaps (The Children) ---
            if children_recaps:
                with st.expander(f"📄 View {len(children_recaps)} Related Follow-ups", expanded=False):
                    for child_id in children_recaps:
                        try:
                            c_row = manager.full_df.loc[child_id]
                            c_date = c_row['parsed_date'].strftime('%b %d') if pd.notna(c_row['parsed_date']) else "???"
                            
                            # Sentiment indicator
                            c_sent_val = float(c_row.get('news_sentiment', 0))
                            c_sent_class = "Positive" if c_sent_val > 0.25 else "Negative" if c_sent_val < -0.25 else "Neutral"
                            c_sent_color = "#4CAF50" if c_sent_class == "Positive" else "#FF5252" if c_sent_class == "Negative" else "#9E9E9E"
                            
                            # Takeaways
                            c_takeaways = str(c_row.get('key_takeaways', '')).split(';')
                            formatted_takeaways = "".join([f"<li style='margin-bottom:2px;'>{t.strip()}</li>" for t in c_takeaways if t.strip()])
                            
                            c_url = c_row.get('url', '#')

                            st.markdown(f"""
                            <div style="padding: 12px; border-left: 4px solid {c_sent_color}; background: #1a1b21; margin-bottom: 12px; border-radius: 6px; border: 1px solid #333;">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                                    <div style="font-size: 0.75rem; color: #aaa;">{c_date} | {c_row.get('source_en', 'Unknown')}</div>
                                    <div style="font-size: 0.75rem; font-weight: bold; color: {c_sent_color}; background: {c_sent_color}1a; padding: 2px 6px; border-radius: 4px;">{c_sent_val:+.2f} {c_sent_class.upper()}</div>
                                </div>
                                <div style="font-size: 0.95rem; font-weight: 600; line-height: 1.3; margin-bottom: 8px;">
                                    <a href="{c_url}" target="_blank" style="text-decoration: none; color: #e0e0e0;">{html.escape(c_row['headline'])}</a>
                                </div>
                                <div style="font-size: 0.8rem; color: #ccc;">
                                    <ul style="margin: 0; padding-left: 15px;">
                                        {formatted_takeaways}
                                    </ul>
                                </div>
                                <div style="margin-top: 8px; text-align: right;">
                                    <a href="{c_url}" target="_blank" style="font-size: 0.7rem; color: #4CAF50; text-decoration: none; font-weight: bold;">VIEW ORIGINAL SOURCE ↗</a>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        except:
                            pass

            if 'search_matches' in row and row['search_matches']:
                 matches_str = " | ".join(row['search_matches'])
                 st.markdown(f"<div style='font-size: 0.75rem; color: #888; margin-top: -8px; margin-bottom: 8px;'>🔍 Matches in: {matches_str}</div>", unsafe_allow_html=True)

        with c_archive:
            is_dup = row.get('is_duplicate_user', False)
            archive_icon = "♻️" if is_dup else "🗑️"
            archive_help = "Restore from Archive" if is_dup else "Mark as Duplicate / Archive"
            
            if st.button(f"{archive_icon}", key=f"archive_btn_{index}", help=archive_help):
                # Quick archive logic: Toggle the duplicate flag
                # Get existing feedback for this item to preserve other fields
                existing_fb = {}
                feedback_df = manager.load_feedback_df()
                if not feedback_df.empty and 'news_id' in feedback_df.columns:
                    match = feedback_df[feedback_df['news_id'] == index].tail(1)
                    if not match.empty:
                        existing_fb = match.iloc[0].to_dict()

                # Basic feedback entry
                feedback_entry = {
                    "news_id": index,
                    "ticker": row['us_ticker_name'],
                    "date": row['parsed_date'],
                    "headline": row['headline'],
                    "user_sentiment_correction": existing_fb.get('user_sentiment_correction', "Correct"),
                    "user_significance_correction": existing_fb.get('user_significance_correction', "Correct"),
                    "source_useful": existing_fb.get('source_useful', "Yes"),
                    "event_correlation_correction": existing_fb.get('event_correlation_correction', "Correct"),
                    "is_duplicate": not is_dup, # TOGGLE
                    "is_recap": existing_fb.get('is_recap', False),
                    "linked_original_id": existing_fb.get('linked_original_id', None),
                    "notes": existing_fb.get('notes', "")
                }
                manager.save_feedback(feedback_entry)
                st.rerun()

        with c_fav:
            if st.button(f"{fav_icon}", key=f"fav_btn_{index}", help="Toggle Favorite"):
                manager.toggle_favorite(index)
                st.rerun()
        
        # Analysis & Stats Columns
        with st.expander("Analysis & Event Lifecycle", expanded=False):
            c1, c2 = st.columns([2, 2])
            
            with c1:
                st.markdown("#### Analysis")
                if pd.notna(row.get('description')):
                    st.markdown(f"**Description:** {row['description']}")

                if pd.notna(row.get('key_takeaways')):
                    takeaways = str(row['key_takeaways']).split(';')
                    formatted_takeaways = "\n".join([f"* {t.strip()}" for t in takeaways if t.strip()])
                    st.markdown(f"**Key Takeaways:**\n\n{formatted_takeaways}")

                st.markdown("")
                
            with c2:
                if market_data:
                    st.markdown("#### Event Lifecycle Analysis")
                    
                    # --- Header: Classification ---
                    h1, h2, h3 = st.columns(3)
                    with h1:
                        st.markdown(f"<div class='metric-label'>Category</div><div class='metric-value-small'>{row.get('event_category', 'N/A')}</div>", unsafe_allow_html=True)
                    with h2:
                        st.markdown(f"<div class='metric-label'>Sentiment</div><span class=\"badge {sentiment_color}\" style='margin-left: 0;'>{sent_cat.upper()} ({row['news_sentiment']:.2f})</span>", unsafe_allow_html=True)                
                        st.markdown("")
                        val = market_data['car_pre_3d']
                        rel_bg = "badge-positive" if val > 0 else "badge-negative"
                        st.markdown(f"<div class='metric-label'>T-3 CAR</div><span class='badge {rel_bg}' style='margin-left: 0;'> {val:.2f}%</span>", unsafe_allow_html=True)
                    with h3:
                        if 'sentiment_alignment' in market_data:
                            align_val = market_data.get('sentiment_alignment', 'N/A')
                            align_color = "metric-positive" if align_val == "Aligned" else "metric-negative" if align_val == "Diverged" else "metric-neutral"
                            
                            pub_context = f"News Published: {market_data.get('timing_label', 'During T+0 Market Hours')}"
                            st.markdown(f"<div class='metric-label'>Sentiment-Price Alignment</div><div class='metric-value-medium'><span class='{align_color}'>{align_val}</span></div><div style='font-size:0.8em; color:#888; margin-top:-4px'></div>", unsafe_allow_html=True)
                        
                        st.markdown("")
                        # Verify if T+3 is actually available based on date
                        days_since = 0
                        if pd.notna(row['parsed_date']):
                             d = row['parsed_date']
                             # Handle both date and timestamp objects
                             if hasattr(d, 'date'):
                                 d = d.date()
                             days_since = (datetime.now().date() - d).days
                        
                        display_label = "T+3 CAR"
                        if 1 <= days_since < 3:
                            display_label = f"T+{days_since} CAR"
                        
                        if days_since >= 1 and 'car_3d' in market_data:
                             val = market_data['car_3d']
                             rel_bg = "badge-positive" if val > 0 else "badge-negative"
                             st.markdown(f"<div class='metric-label'>{display_label}</div><span class='badge {rel_bg}' style='margin-left: 0;'> {val:.2f}%</span>", unsafe_allow_html=True)
                        else:
                             st.markdown(f"<div class='metric-label'>T+3 CAR</div><span class='badge badge-insignificant' style='margin-left: 0;'> N/A</span>", unsafe_allow_html=True)

                    st.markdown('<div style="border-top: 1px solid #444; margin: 10px 0;"></div>', unsafe_allow_html=True)

                    labels = ['T-1', 'T+0', 'T+1']
                    
                    # --- Top Metrics (Visible) ---
                    # Add spacers [0.2] to simulate expander padding alignment
                    top_layout = st.columns([0.2, 10, 10, 10, 0.2])
                    
                    for i, label in enumerate(labels):
                        with top_layout[i+1]: # Use middle columns
                            st.markdown(f"##### {label}")
                            day_metrics = market_data.get(label, {})
                            
                            if not day_metrics:
                                st.markdown("For *No Data*")
                                continue

                            # 1. Stock Close
                            st.markdown(f"<div class='metric-label'>Stock Close</div><div class='metric-value-small'>{day_metrics.get('close', 0):.2f} {delta_span(day_metrics.get('pct_change_day', 0))}</div>", unsafe_allow_html=True)
                            st.markdown("")
                            
                            # 2. Index Close
                            st.markdown(f"<div class='metric-label'>Index Close</div><div class='metric-value-small'>{day_metrics.get('index_close', 0):.2f} {delta_span(day_metrics.get('index_pct_change', 0))}</div>", unsafe_allow_html=True)
                            st.markdown("")

                            # 3. Rel to Index
                            rel = day_metrics.get('relative_change', 0)
                            rel_bg = "badge-positive" if rel > 0 else "badge-negative"
                            st.markdown(f"<div class='metric-label'>Rel. to Index</div><span class='badge {rel_bg}' style='margin-left: 0;'>{rel:+.2f}%</span>", unsafe_allow_html=True)
                            
                    # --- Detailed Metrics (Expandable) ---
                    st.markdown("")
                    with st.expander("Show More Metrics"):
                        det_cols = st.columns(3)
                        for i, label in enumerate(labels):
                            with det_cols[i]:
                                day_metrics = market_data.get(label, {})
                                if not day_metrics: continue # Data check handled in top loop visual, here just skip

                                # 4. Gap %
                                gap = day_metrics.get('gap_pct', 0)
                                prev_c = day_metrics.get('prev_close', 0)
                                open_c = day_metrics.get('open', 0)
                                prev_fmt = f"{prev_c:.0f}" if prev_c > 100 else f"{prev_c:.2f}"
                                open_fmt = f"{open_c:.0f}" if open_c > 100 else f"{open_c:.2f}"
                                st.markdown(f"<div class='metric-label'>Gap % (Market Open)</div><div class='metric-value-small'> <span style='font-size:0.8em; color:#888'>({prev_fmt} -> {open_fmt})</span><br>{delta_span(gap)}</div>", unsafe_allow_html=True)
                                st.markdown("")

                                # 5. Intraday %
                                stock_intra = day_metrics.get('intraday_change', 0)
                                index_intra = day_metrics.get('index_intraday', 0)
                                st.markdown(f"<div class='metric-label'>Intraday (Stock/Idx)</div><div class='metric-value-small'>{delta_span(stock_intra)} / {delta_span(index_intra)}</div>", unsafe_allow_html=True)
                                st.markdown("")

                                # 6. Sigma Move
                                sigma = day_metrics.get('sigma_move', 0)
                                pct = day_metrics.get('pct_change_day', 0)
                                sigma_style = "color: #f59e0b; font-weight: bold;" if abs(sigma) > 2.0 else ""
                                st.markdown(f"<div class='metric-label'>Sigma Move (Last 30d)</div><div class='metric-value-small' style='{sigma_style}'>{sigma:+.1f}σ <span style='font-size:0.8em; color:#888; font-weight:normal'>({pct:+.2f}%)</span></div>", unsafe_allow_html=True)
                                st.markdown("")

                                # 8. Vol Ratio

                                def fmt_vol(v):
                                    if v >= 1_000_000: return f"{v/1_000_000:.1f}M"
                                    elif v >= 1_000: return f"{v/1_000:.1f}K"
                                    return f"{v:.0f}"
                                
                                vol_val = day_metrics.get('volume', 0)
                                vol_str = fmt_vol(vol_val)
                                vol_r = day_metrics.get('volume_rel', 0)
                                vol_r_style = "color: #f59e0b; font-weight: bold;" if vol_r > 1.0 else ""
                                st.markdown(f"<div class='metric-label'>Vol Ratio (Last 20d)</div><div class='metric-value-small' style='{vol_r_style}'>{vol_r:.1f}x <span style='font-size:0.8em; color:#888; font-weight:normal'>({vol_str})</span></div>", unsafe_allow_html=True)                                        
                                st.markdown("")
                else:
                    st.info("Market data unavailable for this date.")


        # Price Chart Expander
        has_daily = market_data and 'chart_data' in market_data and market_data['chart_data'] is not None
        has_hourly = market_data and market_data.get('intraday_data') is not None and not market_data['intraday_data'].empty
        
        with st.expander("Price Trend Analysis", expanded=False):
            if (has_daily or has_hourly):
                # Swap order: Hourly First
                t1, t2 = st.tabs(["Hourly (T-1 to T+1)", "Daily (T-30 to T+15)"])
                
                with t1:
                     if has_hourly:
                        chart_h = charts.create_intraday_chart(
                            market_data['intraday_data'],
                            market_data.get('date_event'),
                            row.get('timestamp'),
                            row.get('parsed_date')
                        )
                        st.altair_chart(chart_h, width="stretch")
                     else:
                        st.info("Hourly data not available (limited to last 730 days).")

                with t2:
                    if has_daily:
                        chart_d = charts.create_daily_chart(market_data)
                        st.altair_chart(chart_d, width="stretch")
                    else:
                        st.info("Daily chart data unavailable.")
            else:
                st.info("Market data unavailable for this date range.")
        
        # Model Reasoning Expander
        with st.expander("Model Reasoning", expanded=False):
            if pd.notna(row.get('actionable_intelligence')):
                st.markdown("##### Actionable Intelligence")
                st.markdown(row['actionable_intelligence'])
                st.markdown("---")

            if pd.notna(row.get('breaking_recap_reasoning')):
                st.markdown("##### Breaking Recap Reasoning")
                st.markdown(row['breaking_recap_reasoning'])
                st.markdown("---")

            if pd.notna(row.get('news_worthy_reasoning')):
                st.markdown("##### News-worthy Reasoning")
                st.markdown(row['news_worthy_reasoning'])

        # Feedback / Model Validation Form
        with st.expander("Model Validation", expanded=False):
            f1, f2 = st.columns(2)
            with f1:
                st.caption(f"Sentiment Correction **{row['sentiment_category']}**")
                sent_correct = st.radio("Is the assigned sentiment accurate?", ["Yes", "No"], horizontal=True, key=f"sent_check_{index}")
                
                final_sentiment = row['news_sentiment']
                if sent_correct == "No":
                    final_sentiment = st.slider("Correct Sentiment", -1.0, 1.0, 0.0, 0.1, key=f"sent_fix_{index}")

            with f2:
                st.caption("Alignment Validation")
                corr_correct = st.radio("Does the sentiment correctly reflect the price movement?", ["Yes", "No"], horizontal=True, key=f"corr_check_{index}")
                
                final_corr = "Correct"
                if corr_correct == "No":
                    final_corr = st.selectbox("Select Correct Alignment", ["Aligned", "Divergent", "Unknown"], key=f"corr_fix_{index}")

            f3, f4 = st.columns(2)
            
            with f3:
                st.caption("Source Quality")
                source_useful = st.radio("Is the source or reporter reliable?", ["Yes", "No"], horizontal=True, key=f"src_check_{index}")

            with f4:
                st.caption(f"News Significance: **{row['classification']}**")
                sig_correct = st.radio("Is this event significant?", ["Yes", "No"], horizontal=True, key=f"sig_check_{index}")
                
                final_sig = row['classification']
                if sig_correct == "No":
                    sig_option = st.selectbox("Select Correct Significance", ["HIGH", "LOW"], key=f"sig_fix_{index}")
                    final_sig = "SIGNIFICANT" if sig_option == "HIGH" else "INSIGNIFICANT"

            st.caption("Additional Feedback")
            f_recap = st.checkbox("Mark this news as a recap story", key=f"recap_{index}")
            
            linked_original_id = None
            if f_recap:
                    if not manager.full_df.empty:
                        ticker = row['us_ticker_name']
                        date = row['parsed_date']
                        
                        # Toggle: Search only this ticker OR all tickers?
                        search_all_tickers = st.checkbox("🔍 Search all tickers (not just this one)", key=f"cross_tick_{index}")
                        
                        # Search candidates: same ticker OR all (based on toggle), <= date, excluding itself
                        if search_all_tickers:
                            # Limited only by date and excluding itself
                            mask = (manager.full_df['parsed_date'] <= date) & (manager.full_df.index != index)
                        else:
                            # Strict: same ticker AND date constrained
                            mask = (manager.full_df['us_ticker_name'] == ticker) & \
                                   (manager.full_df['parsed_date'] <= date) & \
                                   (manager.full_df.index != index)
                        
                        # 2. Sort candidates CHRONOLOGICALLY (Newest First)
                        temp_candidates = manager.full_df[mask].copy()
                        
                        def get_sort_dt(r):
                            d = r.get('parsed_date')
                            if pd.isna(d): return pd.Timestamp.min
                            ts = r.get('timestamp')
                            if pd.notna(ts) and str(ts).strip():
                                try:
                                    t_obj = pd.to_datetime(str(ts), errors='coerce')
                                    if pd.notna(t_obj): return datetime.combine(d, t_obj.time())
                                except: pass
                            return datetime.combine(d, datetime.min.time())

                        temp_candidates['_sort_dt'] = temp_candidates.apply(get_sort_dt, axis=1)
                        candidates = temp_candidates.sort_values('_sort_dt', ascending=False)
                        
                        if not candidates.empty:
                            def format_label(idx):
                                if idx == "None": return "Showing all (use filter below to narrow)..."
                                c_row = manager.full_df.loc[idx]
                                d_str = c_row['parsed_date'].strftime('%b %d') if pd.notna(c_row['parsed_date']) else "???"
                                
                                # Badge: [TICKER] if search_all is on
                                tick_label = f"[{c_row['us_ticker_name']}] " if search_all_tickers else ""
                                
                                t_str = ""
                                c_ts = c_row.get('timestamp')
                                if pd.notna(c_ts) and str(c_ts).strip():
                                    try:
                                        t_obj = pd.to_datetime(str(c_ts), errors='coerce')
                                        if pd.notna(t_obj): t_str = f" {t_obj.strftime('%I:%M %p')}"
                                    except: pass
                                
                                type_label = str(c_row.get('breaking_recap', 'UNK')).upper()
                                return f"[{d_str}{t_str}] {tick_label}[{type_label}] {c_row['headline']}"

                        # Stage 1: Filter keywords manually to keep the sort order stable
                        search_q = st.text_input("🔍 Filter Original Stories by keyword", key=f"link_search_{index}", help="Type to narrow down headlines while keeping chronological order.")
                        
                        filtered_candidates = candidates
                        if search_q:
                            filtered_candidates = candidates[candidates['headline'].str.contains(search_q, case=False, na=False)]

                        # Stage 2: The Selectbox (Now strictly sorted, no internal jumpiness)
                        linked_original_id = st.selectbox(
                            "Select the original story event...",
                            options=["None"] + filtered_candidates.index.tolist(),
                            format_func=format_label,
                            key=f"link_orig_{index}",
                            help="Select the breaking news event this recap refers to."
                        )
                        if linked_original_id == "None":
                            linked_original_id = None

            f_notes = st.text_area("Notes", placeholder="Reasoning for correction...", height=80, key=f"note_{index}")
            
            if st.button("Submit Feedback", key=f"btn_{index}"):
                feedback_entry = {
                    "news_id": index,
                    "ticker": row['us_ticker_name'],
                    "date": row['parsed_date'],
                    "headline": row['headline'],
                    "user_sentiment_correction": final_sentiment if sent_correct == "No" else "Correct",
                    "user_significance_correction": final_sig if sig_correct == "No" else "Correct",
                    "source_useful": source_useful,
                    "event_correlation_correction": final_corr,
                    "is_duplicate": row.get('is_duplicate_user', False),
                    "is_recap": f_recap,
                    "linked_original_id": linked_original_id,
                    "notes": f_notes
                }
                manager.save_feedback(feedback_entry)
                st.success(f"Feedback saved! {'(Linked to Original)' if linked_original_id is not None else ''}")

        with st.expander("Show Source Details"):
            st.markdown(f"**Original Headline:** {row['original_headline']}")
            st.markdown(f"**Link:** {row['url']}")
            
    st.markdown("---")
