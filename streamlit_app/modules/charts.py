import altair as alt
import pandas as pd
from datetime import datetime, timedelta
import pytz

def create_intraday_chart(intraday_data, event_date_obj, news_time_str, parsed_date):
    """
    Creates an Altair chart for intraday price movement with news marker and T-1/T+1 boundaries.
    """
    # Prepare Data
    h_df = intraday_data.reset_index()
    # Rename first column usually Datetime
    h_df.rename(columns={h_df.columns[0]: 'Time', 'Close': 'Price'}, inplace=True)

    # Fix for Altair Timezone (Strip TZ after ensuring JST)
    if pd.api.types.is_datetime64_any_dtype(h_df['Time']):
        if h_df['Time'].dt.tz is not None:
            try:
                h_df['Time'] = h_df['Time'].dt.tz_convert('Asia/Tokyo').dt.tz_localize(None)
            except:
                h_df['Time'] = h_df['Time'].dt.tz_localize(None)

    # Default Domains based on data with buffer
    min_time = h_df['Time'].min() - timedelta(hours=5)
    max_time = h_df['Time'].max() + timedelta(hours=5)
    
    # --- Logic to add News Marker ---
    news_dt_jst = None
    
    if pd.notna(news_time_str) and str(news_time_str).strip() != '':
        try:
            # Anchor to parsed_date (Event US Date) or today if missing
            row_date = parsed_date if parsed_date else datetime.now().date()
            
            # Parse time string
            time_part = pd.to_datetime(str(news_time_str), format='%H:%M', errors='coerce').time()
            # If format failed, maybe try parsing full date string? 
            # The original code used pd.to_datetime(str(ts_val), format='%H:%M').
            # We stick to original logic for compatibility.
            
            if time_part:
                naive_dt = datetime.combine(row_date, time_part)
                eastern = pytz.timezone('America/New_York')
                us_dt = eastern.localize(naive_dt)
                jst = pytz.timezone('Asia/Tokyo')
                converted_jst = us_dt.astimezone(jst)
                # Make Naive for Altair
                converted_jst_naive = converted_jst.replace(tzinfo=None)
                news_dt_jst = converted_jst_naive
        except Exception as e:
            pass

    # Calculate Final Domain including Marker
    final_min_time = min_time
    final_max_time = max_time
    
    if news_dt_jst:
        # Check against existing data bounds (ensure chart doesn't break if valid/invalid)
        if pd.notna(final_min_time) and news_dt_jst < final_min_time:
            final_min_time = news_dt_jst - timedelta(hours=1)
        if pd.notna(final_max_time) and news_dt_jst > final_max_time:
            final_max_time = news_dt_jst + timedelta(hours=1)
        
        # Ensure we can see it if it's close to edge
        if pd.notna(final_max_time) and (final_max_time - news_dt_jst).total_seconds() < 3600:
            final_max_time = final_max_time + timedelta(hours=1)

    
    # Calculate Price Domain with Buffer (10%)
    p_min = h_df['Price'].min()
    p_max = h_df['Price'].max()
    p_diff = p_max - p_min
    if p_diff == 0: p_diff = p_max * 0.01 # Handle flatline
    
    y_domain = [p_min - (p_diff * 0.2), p_max + (p_diff * 0.2)]

    # Base Chart
    base_h = alt.Chart(h_df).encode(
        x=alt.X('Time:T', 
                axis=alt.Axis(format='%m/%d %H:%M', title='Time (JST)'),
                scale=alt.Scale(domain=[final_min_time, final_max_time])
        ),
        y=alt.Y('Price:Q', scale=alt.Scale(domain=y_domain), title='Price (JPY)')
    )
    line_h = base_h.mark_line(point=True, color='#3b82f6').encode(
        tooltip=[
            alt.Tooltip('Time:T', format='%m/%d %H:%M', title='Time'),
            alt.Tooltip('Price:Q', format='.2f', title='Price')
        ]
    )
    
    chart_layers = [line_h]
    
    # --- Add Date Boundary Lines (Red) ---
    unique_dates = sorted(h_df['Time'].dt.date.unique())
    
    # Find index of Event Date to calculate Trading Day Offsets
    event_idx = -1
    # Check if event_date_obj is present
    # event_date_obj might be datetime or date/timestamp
    if event_date_obj:
         evt_d = event_date_obj.date() if hasattr(event_date_obj, 'date') else event_date_obj
         if evt_d in unique_dates:
             event_idx = list(unique_dates).index(evt_d)

    day_rules_data = []
    for i, d in enumerate(unique_dates):
        start_time = h_df[h_df['Time'].dt.date == d]['Time'].min()
        
        label = f"{d.strftime('%m/%d')}"
        if event_idx != -1:
            offset = i - event_idx
            if offset == 0: label = f"T+0 ({d.strftime('%m/%d')})"
            elif offset > 0: label = f"T+{offset} ({d.strftime('%m/%d')})"
            elif offset < 0: label = f"T{offset} ({d.strftime('%m/%d')})"
        
        day_rules_data.append({'Time': start_time, 'Label': label, 'Color': 'red'})
        
    if day_rules_data:
        day_rules_df = pd.DataFrame(day_rules_data)
        day_rules = alt.Chart(day_rules_df).mark_rule(strokeDash=[2, 2], opacity=0.6).encode(
            x='Time:T',
            color=alt.Color('Color:N', scale=None)
        )
        day_labels = alt.Chart(day_rules_df).mark_text(
            align='left', dx=5, dy=20, baseline='top', fontSize=12, color='red'
        ).encode(
            x='Time:T',
            y=alt.value(0), # Place at Top
            text='Label:N'
        )

        chart_layers.append(day_rules)
        chart_layers.append(day_labels)

    # Add Marker if available
    if news_dt_jst:
        # Ensure news_dt_jst is naive
        if news_dt_jst.tzinfo is not None:
            news_dt_jst = news_dt_jst.replace(tzinfo=None)

        final_marker_time = news_dt_jst
        
        rule_df = pd.DataFrame({
            'Time': [final_marker_time], 
            'Label': ['News Published']
        })
        
        # Vertical Line (Yellow)
        news_rule = alt.Chart(rule_df).mark_rule(color='yellow', strokeWidth=2).encode(x='Time:T')
                
        # Label (Bottom of line)
        news_label = alt.Chart(rule_df).mark_text(
            align='left',
            baseline='top',
            dx=5,  # Shift right 5 pixels
            dy=0, # Shift down
            color='yellow', 
            fontWeight='bold',
            fontSize=12
        ).encode(
            x='Time:T',      # Horizontal position based on Data Time
            y=alt.value(0),  # Vertical position fixed at 0 pixels (Top)
            text='Label:N'
        )
                
        chart_layers.append(news_rule)
        chart_layers.append(news_label)

    combined_chart = alt.layer(*chart_layers).properties(height=400).interactive()
    return combined_chart


def create_daily_chart(market_data):
    """
    Creates an Altair chart for daily price metrics including CAR labels.
    """
    chart_df = market_data['chart_data'].reset_index()
    # Rename first column
    chart_df.rename(columns={chart_df.columns[0]: 'Date'}, inplace=True)
    chart_df['Date'] = pd.to_datetime(chart_df['Date'])
    
    # Melt for Altair
    chart_df = chart_df.melt('Date', var_name='Asset', value_name='Return')
    
    # Calculate Return Domain with Buffer (50%)
    r_min = chart_df['Return'].min()
    r_max = chart_df['Return'].max()
    r_diff = r_max - r_min
    if r_diff == 0: r_diff = 1.0 # Handle flatline
    
    y_domain = [r_min - (r_diff * 0.2), r_max + (r_diff * 0.2)]
    
    # Create Chart (Daily)
    base = alt.Chart(chart_df).encode(
        x=alt.X('Date:T', axis=alt.Axis(format='%m/%d/%y', title='Date'))
    )

    lines = base.mark_line(point=True).encode(
        y=alt.Y('Return:Q', scale=alt.Scale(domain=y_domain), axis=alt.Axis(format='.0f'), title='Return (%)'),
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
         val_str = f"{market_data.get('car_pre_3d', 0):.2f}%"
         rules_data.append({'Date': pd.Timestamp(market_data['date_t_minus_3']), 'Color': 'red', 'Label': f'T-3 CAR ({val_str})'})
    if market_data.get('date_t_plus_3'):
         val_str = f"{market_data.get('car_3d', 0):.2f}%"
         rules_data.append({'Date': pd.Timestamp(market_data['date_t_plus_3']), 'Color': 'red', 'Label': f'T+3 CAR ({val_str})'})
    rules_df = pd.DataFrame(rules_data)
    
    chart_layers = [lines]

    if not rules_df.empty:
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
            dy=20,
            fontSize=12
        ).encode(
            x='Date:T',
            y=alt.value(0),
            text='Label:N',
            color=alt.Color('Color:N', scale=None)
        )
        chart_layers.append(rules)
        chart_layers.append(rule_labels)
    
    # Highlight Event Day (Vertical Line)
    event_date = market_data.get('date_event')
    if event_date:
        # Create a dataframe just for the rule
        e_rule_df = pd.DataFrame([{'Date': pd.Timestamp(event_date), 'Label': 'Event Day'}])
        
        event_rule = alt.Chart(e_rule_df).mark_rule(color='yellow', strokeWidth=2).encode(
            x='Date:T'
        )
        
        # Event Day Label
        event_label = alt.Chart(e_rule_df).mark_text(
            align='left',
            baseline='top',
            dx=5,
            dy=0, 
            fontSize=12,
            fontWeight='bold',
            color='yellow'
        ).encode(
            x='Date:T',
            y=alt.value(0),
            text='Label:N'
        )
        chart_layers.append(event_rule)
        chart_layers.append(event_label)
            
    combined_chart = alt.layer(*chart_layers).properties(height=400).interactive()
    return combined_chart
