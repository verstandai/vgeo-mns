import streamlit as st
import pandas as pd
import html
import textwrap
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
        st.metric("Total Aligned Events", len(view_df))
    with col2:
        st.metric("Bullish Aligned Events", bullish_count)
    with col3:
        st.metric("Bearish Aligned Events", bearish_count)
    st.markdown("---")

def render_news_card(index, row, manager):
    """Renders a single news card with details, feedback form, and lifecycle grid."""
    
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

    # Badges
    sig_badge = "badge-significant" if row['classification'].upper() == 'SIGNIFICANT' else "badge-insignificant"
    recap_badge = "badge-breaking" if row['breaking_recap'].upper() == 'BREAKING' else "badge-insignificant" if row['breaking_recap'] == 'Recap' else "badge-insignificant"

    # Format Date & Time
    date_str = row['parsed_date'].strftime('%B %d, %Y') if row['parsed_date'] else row['date']
    
    time_str = ""
    ts_val = row.get('timestamp')
    if pd.notna(ts_val) and str(ts_val).strip() != '':
        try:
            ts_obj = pd.to_datetime(str(ts_val), format='%H:%M', errors='coerce')
            if pd.notna(ts_obj):
                curr_date = datetime.now().date() 
                naive_dt = datetime.combine(curr_date, ts_obj.time())
                eastern = pytz.timezone('America/New_York')
                est_dt = eastern.localize(naive_dt)
                jst = pytz.timezone('Asia/Tokyo')
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
    
    safe_headline = html.escape(str(row['headline']))
    safe_company = html.escape(str(row.get('company_name', '')))
    safe_source = html.escape(str(row['source_en']))
    safe_reporter = html.escape(str(row.get('reporter', 'Unknown')))

    with st.container():
        # Header Row
        # Card (Left, Large) | Button (Right, Small)
        c_head, c_fav = st.columns([20, 1])
        
        with c_head:
            recap_badge_html = f'<span class="badge {recap_badge}">{row.get("breaking_recap", "Unknown").upper()}</span>'
            sig_badge_html = f'<span class="badge {sig_badge}">{row.get("classification", "Unknown").upper()}</span>'

            st.markdown(textwrap.dedent(f"""
                <div class="news-card">
                <div class="card-header" style="margin-bottom: 4px;">
                <div class="header-left">
                <span class="news-date">{display_date}</span>
                </div>
                <div>
                {recap_badge_html}
                {sig_badge_html}
                </div>
                </div>
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
                <span class="ticker-badge" style="margin-left: 0;">{row['us_ticker_name']}</span>
                <span class="company-name">{safe_company}</span>
                </div>
                <div class="news-headline">{safe_headline}</div>
                <div class="news-source">Source: {safe_source} | Reporter: {safe_reporter}</div>
                </div>
                """), unsafe_allow_html=True)

        with c_fav:
            if st.button(f"{fav_icon}", key=f"fav_btn_{index}", help="Toggle Favorite"):
                manager.toggle_favorite(index)
                st.rerun()
        
        # Analysis & Stats Columns
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
                    rel_bg = "badge-positive" if market_data.get('car_3d', 0) > 0 else "badge-negative"
                    st.markdown(f"<div class='metric-label'>T-3 CAR</div><span class='badge {rel_bg}' style='margin-left: 0;'> {market_data.get('car_pre_3d', 'N/A'):.2f}%</span>", unsafe_allow_html=True)
                with h3:
                    if 'sentiment_alignment' in market_data:
                        align_val = market_data.get('sentiment_alignment', 'N/A')
                        align_color = "metric-positive" if align_val == "Aligned" else "metric-negative" if align_val == "Diverged" else "metric-neutral"
                        
                        pub_context = f"News Published: {market_data.get('timing_label', 'During T+0 Market Hours')}"
                        st.markdown(f"<div class='metric-label'>Sentiment-Price Alignment</div><div class='metric-value-medium'><span class='{align_color}'>{align_val}</span></div><div style='font-size:0.8em; color:#888; margin-top:-4px'></div>", unsafe_allow_html=True)
                    
                    st.markdown("")
                    rel_bg = "badge-positive" if market_data.get('car_3d', 0) > 0 else "badge-negative"
                    st.markdown(f"<div class='metric-label'>T+3 CAR</div><span class='badge {rel_bg}' style='margin-left: 0;'> {market_data.get('car_3d', 'N/A'):.2f}%</span>", unsafe_allow_html=True)

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
                            vol_val = day_metrics.get('volume', 0)
                            def fmt_vol(v):
                                if v >= 1_000_000: return f"{v/1_000_000:.1f}M"
                                elif v >= 1_000: return f"{v/1_000:.1f}K"
                                return f"{v:.0f}"
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
        
        if (has_daily or has_hourly):
            with st.expander("Price Trend Analysis", expanded=False):
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
                        st.altair_chart(chart_h, width='stretch')
                     else:
                        st.info("Hourly data not available (limited to last 730 days).")

                with t2:
                    if has_daily:
                        chart_d = charts.create_daily_chart(market_data)
                        st.altair_chart(chart_d, width='stretch')
                    else:
                        st.info("Daily chart data unavailable.")
        
        # Model Reasoning Expander
        with st.expander("Model Reasoning", expanded=False):
            if pd.notna(row.get('actionable_intelligence')):
                st.expander("Actionable Intelligence", expanded=True).markdown(row['actionable_intelligence'])

            if pd.notna(row.get('breaking_recap_reasoning')):
                st.expander("Breaking Recap Reasoning", expanded=True).markdown(row['breaking_recap_reasoning'])

            if pd.notna(row.get('news_worthy_reasoning')):
                st.expander("News-worthy Reasoning", expanded=True).markdown(row['news_worthy_reasoning'])

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
            f_dup = st.checkbox("Mark this news as a duplicate", key=f"dup_{index}")
            f_recap = st.checkbox("Mark this news as a recap story", key=f"recap_{index}")
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
