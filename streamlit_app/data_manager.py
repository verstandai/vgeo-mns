import pandas as pd
import yfinance as yf
import os
import json
import numpy as np
from datetime import datetime, timedelta

class DataManager:
    def __init__(self, data_path, feedback_path):
        self.data_path = data_path
        self.feedback_path = feedback_path
        self.market_data_cache = {}
        # Initialize Favorites
        self.favorites_file = os.path.join(os.path.dirname(__file__), 'favorites.json')
        self.favorites = self.load_favorites()

    def load_data(self):
        """Loads the CSV data."""
        if not os.path.exists(self.data_path):
            return pd.DataFrame()
        
        # 1. Load Raw Data
        df = pd.read_csv(self.data_path)
        
        # 2. Parse Dates (Handle 11-Nov -> 2025-11-11)
        def parse_date(d):
            if pd.isna(d): return pd.NaT
            s = str(d).strip()
            # Try MM/DD/YY (Old)
            try: return datetime.strptime(s, '%m/%d/%y').date()
            except: pass
            # Try DD-Mon -> Assume 2025
            try: return datetime.strptime(f"{s}-2025", '%d-%b-%Y').date()
            except: pass
            return pd.NaT

        df['parsed_date'] = df['date'].apply(parse_date)
        
        # 3. Convert 'date' column to desired '11/17/2025' string format
        df['date'] = df['parsed_date'].apply(lambda x: x.strftime('%m/%d/%Y') if pd.notnull(x) else '')

        # Sort by Date (Desc)
        if 'parsed_date' in df.columns:
            df = df.sort_values(by=['parsed_date'], ascending=False)

        return df

    def load_favorites(self):
        """Loads the list of favorite news IDs."""
        if not os.path.exists(self.favorites_file):
            return []
        try:
            with open(self.favorites_file, 'r') as f:
                return json.load(f)
        except:
            return []

    def save_favorites(self):
        """Saves the favorites list to JSON."""
        with open(self.favorites_file, 'w') as f:
            json.dump(self.favorites, f)

    def toggle_favorite(self, news_id):
        """Toggles the favorite status of a news item."""
        # Ensure news_id is treated consistently (e.g., as int or string)
        # Using index from dataframe which is usually int
        if news_id in self.favorites:
            self.favorites.remove(news_id)
        else:
            self.favorites.append(news_id)
        self.save_favorites()
        return news_id in self.favorites

    def is_favorite(self, news_id):
        return news_id in self.favorites

    def get_market_data(self, ticker, date_obj, local_ticker=None, index_name=None, sentiment_score=None, news_time_str=None):
        """
        Fetches market data for a specific ticker and date.
        Returns a dictionary of metrics including CAR and Impact analysis.
        """
        if date_obj is None or pd.isna(date_obj):
            return None

        # Create a cache key
        key = f"{ticker}_{local_ticker}_{date_obj}_{sentiment_score}_v7"
        if key in self.market_data_cache:
            return self.market_data_cache[key]

        metrics = {}
        
        try:
            # Determine Ticker Symbol
            target_ticker = ticker
            if local_ticker:
                if str(local_ticker).isdigit():
                    target_ticker = f"{local_ticker}.T"
                else:
                    target_ticker = local_ticker
            
            # Determine Index Ticker
            index_ticker = "^TOPX" # Default
            if index_name == "TOPIX":
                index_ticker = "1306.T" # Use ETF as proxy
            
            # Define window: Start = Date - 30 days (for volatility/avg), End = Date + 15 days for buffer
            # Convert to string to ensure yfinance handles it correctly
            start_date = (date_obj - timedelta(days=30)).strftime('%Y-%m-%d')
            end_date = (date_obj + timedelta(days=15)).strftime('%Y-%m-%d')
            
            # Fetch Stock Data
            stock = yf.Ticker(target_ticker)
            hist = stock.history(start=start_date, end=end_date)
            
            # Fetch Index Data
            # Try ^TOPX first
            idx = yf.Ticker(index_ticker)
            hist_idx = idx.history(start=start_date, end=end_date)
            
            # Fallback to 1306.T (TOPIX ETF) if ^TOPX is empty or missing
            if hist_idx.empty and index_ticker == "^TOPX":
                print("⚠️ ^TOPX data missing, falling back to 1306.T")
                idx = yf.Ticker("1306.T")
                hist_idx = idx.history(start=start_date, end=end_date)

            if hist.empty:
                raise ValueError(f"No data found for {target_ticker}")

            # Normalize index to date only and sort
            hist.index = hist.index.date
            hist = hist.sort_index()
            
            if not hist_idx.empty:
                hist_idx.index = hist_idx.index.date
                hist_idx = hist_idx.sort_index()
            
            # Locate the specific date
            target_ts = date_obj
            
            # Find nearest trading day if needed
            if target_ts not in hist.index:
                future_days = [d for d in hist.index if d >= target_ts]
                if future_days:
                    target_ts = future_days[0]
                else:
                    target_ts = hist.index[-1]

            # --- Iterate for T-1, T+0, T+1 ---
            days_to_calc = {'T-1': -1, 'T+0': 0, 'T+1': 1}
            
            # Get the integer location of target_ts in the history index
            loc = hist.index.get_loc(target_ts)

            for label, offset in days_to_calc.items():
                day_metrics = {}
                # Default values
                day_metrics['close'] = 0.0
                day_metrics['pct_change_day'] = 0.0
                day_metrics['gap_pct'] = 0.0
                day_metrics['intraday_change'] = 0.0
                day_metrics['sigma_move'] = 0.0
                day_metrics['volume_rel'] = 0.0
                day_metrics['volume_z_score'] = 0.0
                day_metrics['index_close'] = 0.0
                day_metrics['index_pct_change'] = 0.0
                day_metrics['relative_change'] = 0.0
                day_metrics['prev_close'] = 0.0
                day_metrics['open'] = 0.0
                
                # Check bounds
                if 0 <= loc + offset < len(hist):
                    current_ts = hist.index[loc + offset]
                    day_data = hist.loc[current_ts]
                    
                    # Previous Close (for Gap and Pct Change)
                    prev_close = day_data['Open'] # Default
                    if loc + offset > 0:
                        prev_close = hist.iloc[loc + offset - 1]['Close']
                    day_metrics['prev_close'] = prev_close

                    # Basic Metrics
                    day_metrics['close'] = day_data['Close']
                    day_metrics['open'] = day_data['Open']
                    day_metrics['pct_change_day'] = ((day_data['Close'] - prev_close) / prev_close) * 100
                    day_metrics['volume'] = day_data['Volume']
                    
                    # Gap %
                    day_metrics['gap_pct'] = ((day_data['Open'] - prev_close) / prev_close) * 100
                    
                    # Intraday
                    if day_data['Open'] and day_data['Open'] > 0:
                        day_metrics['intraday_change'] = ((day_data['Close'] - day_data['Open']) / day_data['Open']) * 100
                    
                    # Sigma (Volatility)
                    # Window: 30 days prior to THIS specific day
                    past_hist = hist.loc[:current_ts].iloc[:-1] 
                    if len(past_hist) >= 20:
                        past_returns = past_hist['Close'].pct_change().dropna().tail(30)
                        mean_ret = past_returns.mean()
                        std_ret = past_returns.std()
                        day_ret = day_metrics['pct_change_day'] / 100.0
                        if std_ret > 0:
                            day_metrics['sigma_move'] = (day_ret - mean_ret) / std_ret
                    
                    # Volume Metrics
                    # Window: 20 days prior to THIS specific day
                    window_hist = hist.loc[:current_ts].iloc[-21:-1]
                    if not window_hist.empty:
                        avg_vol = window_hist['Volume'].mean()
                        std_vol = window_hist['Volume'].std()
                        day_metrics['volume_rel'] = day_metrics['volume'] / avg_vol if avg_vol else 1.0
                        day_metrics['volume_z_score'] = (day_metrics['volume'] - avg_vol) / std_vol if std_vol else 0.0
                    else:
                        day_metrics['volume_rel'] = 1.0
                    
                    # Index Metrics for this day
                    if not hist_idx.empty and current_ts in hist_idx.index:
                        idx_day = hist_idx.loc[current_ts]
                        
                        # Handle potential duplicate dates (force to Series)
                        if isinstance(idx_day, pd.DataFrame):
                            idx_day = idx_day.iloc[0]
                            
                        day_metrics['index_close'] = idx_day['Close']
                        
                        # Index Intraday
                        # Check Open validity carefully
                        i_open = idx_day.get('Open', 0)
                        if pd.notna(i_open) and i_open > 0:
                            day_metrics['index_intraday'] = ((idx_day['Close'] - i_open) / i_open) * 100
                            
                        # Index Pct Change
                        # Need prev index close
                        idx_loc = hist_idx.index.get_loc(current_ts)
                        
                        # get_loc might return slice/array if duplicates. Handle it.
                        if isinstance(idx_loc, slice):
                            idx_loc = idx_loc.start
                        elif hasattr(idx_loc, '__iter__'):
                             # numpy array or list
                             idx_loc = idx_loc[0]
                             
                        if idx_loc > 0:
                            prev_idx_close = hist_idx.iloc[idx_loc - 1]['Close']
                            day_metrics['index_pct_change'] = ((idx_day['Close'] - prev_idx_close) / prev_idx_close) * 100
                            
                    # Relative Change
                    day_metrics['relative_change'] = day_metrics['pct_change_day'] - day_metrics['index_pct_change']
                
                metrics[label] = day_metrics

            # --- CAR Calculation (T+1 to T+3) ---
            # We want strictly post-event trend
            car = 0.0
            # Get dates AFTER target_ts
            # Slice [1:4] gets T+1, T+2, T+3 (index 0 is target_ts, so next is T+1? No, logic above: d >= target_ts)
            # post_dates list: [T+0, T+1, T+2, T+3, ...]
            all_post = [d for d in hist.index if d >= target_ts]
            
            # T+0 AR
            car_t0 = 0.0
            if len(all_post) > 0:
                t0_date = all_post[0]
                d_data = hist.loc[t0_date]
                d_loc = hist.index.get_loc(t0_date)
                d_prev = hist.iloc[d_loc - 1]['Close'] if d_loc > 0 else d_data['Open']
                s_ret = (d_data['Close'] - d_prev) / d_prev
                
                i_ret = 0.0
                if not hist_idx.empty and t0_date in hist_idx.index:
                    i_data = hist_idx.loc[t0_date]
                    i_loc = hist_idx.index.get_loc(t0_date)
                    i_prev = hist_idx.iloc[i_loc - 1]['Close'] if i_loc > 0 else i_data['Open']
                    i_ret = (i_data['Close'] - i_prev) / i_prev
                car_t0 = s_ret - i_ret

            # T+1 to T+3 AR
            post_dates = all_post[1:4] # Take next 3 days
            
            for i, d in enumerate(post_dates):
                # Stock Return
                d_data = hist.loc[d]
                # Find prev close for daily return
                d_loc = hist.index.get_loc(d)
                d_prev = hist.iloc[d_loc - 1]['Close'] if d_loc > 0 else d_data['Open']
                s_ret = (d_data['Close'] - d_prev) / d_prev
                
                # Index Return
                i_ret = 0.0
                if not hist_idx.empty and d in hist_idx.index:
                    i_data = hist_idx.loc[d]
                    i_loc = hist_idx.index.get_loc(d)
                    i_prev = hist_idx.iloc[i_loc - 1]['Close'] if i_loc > 0 else i_data['Open']
                    i_ret = (i_data['Close'] - i_prev) / i_prev
                
                daily_ar = (s_ret - i_ret)
                car += daily_ar
            
            metrics['car_3d'] = car * 100 # Convert to percentage
            metrics['abnormal_return'] = car_t0 * 100
            
            # --- Pre-Event CAR Calculation (T-3 to T-1) ---
            car_pre = 0.0
            # Get dates strictly before target_ts
            pre_dates = [d for d in hist.index if d < target_ts][-3:] # Last 3 days before event
            
            for d in pre_dates:
                # Stock Return
                d_data = hist.loc[d]
                # Previous Close logic
                d_loc = hist.index.get_loc(d)
                d_prev = hist.iloc[d_loc - 1]['Close'] if d_loc > 0 else d_data['Open']
                s_ret = (d_data['Close'] - d_prev) / d_prev
                
                # Index Return
                i_ret = 0.0
                if not hist_idx.empty and d in hist_idx.index:
                    i_data = hist_idx.loc[d]
                    i_loc = hist_idx.index.get_loc(d)
                    i_prev = hist_idx.iloc[i_loc - 1]['Close'] if i_loc > 0 else i_data['Open']
                    i_ret = (i_data['Close'] - i_prev) / i_prev
                
                car_pre += (s_ret - i_ret)
            
            metrics['car_pre_3d'] = car_pre * 100
            
            # Store Key Dates for Charting
            metrics['date_event'] = target_ts
            metrics['date_t_minus_3'] = pre_dates[0] if pre_dates else None
            # Store T+3 date correctly (it's the last one in post_dates or if list is empty use something else)
            metrics['date_t_plus_3'] = post_dates[-1] if post_dates else (all_post[-1] if all_post else None)

            # Fetch Intraday (Hourly) Data for the event day range (T-1 to T+1 Trading Days)
            # Note: yfinance only provides 1h data for the last 730 days
            metrics['intraday_data'] = None
            try:
                days_diff = (datetime.now().date() - target_ts).days
                if days_diff < 729:
                    # Leverage existing 'hist' which contains daily trading data
                    # Find location of target_ts in hist
                    if target_ts in hist.index:
                        loc = hist.index.get_loc(target_ts)
                        # We want T-1 (loc-1) to T+1 (loc+1)
                        # Ensure bounds
                        start_idx = max(0, loc - 1)
                        end_idx = min(len(hist) - 1, loc + 2) # we want up to T+1, so index T+2 for exclusive slicing or get T+2 date
                        
                        # Get actual dates
                        # start_date is T-1
                        t_minus_1_date = hist.index[start_idx]
                        
                        # end_date for yfinance fetch must be T+2 (exclusive) to include T+1
                        # If T+2 is out of bounds (future), we just use the next available day or today + 1
                        if loc + 2 < len(hist):
                            t_plus_2_date = hist.index[loc + 2]
                        else:
                            # If we are at the end variance, just add plenty of calendar days to ensure coverage
                            t_plus_2_date = hist.index[-1] + timedelta(days=5)

                        intra_start = t_minus_1_date.strftime('%Y-%m-%d')
                        intra_end = t_plus_2_date.strftime('%Y-%m-%d')
                        
                        intra_df = stock.history(start=intra_start, end=intra_end, interval="1h")
                        if not intra_df.empty:
                            metrics['intraday_data'] = intra_df
                    else:
                        # Fallback if target_ts missing (e.g. holiday event mapped to next day?)
                        # Try naive
                         pass
            except Exception as e:
                print(f"Intraday fetch failed: {e}")

            # Prepare Chart Data – daily returns (day‑over‑day)
            df_chart = pd.DataFrame({
                'Stock': hist['Close'],
                'Index': hist_idx['Close'] if not hist_idx.empty else None
            })
            # Forward fill missing data and drop NaNs
            df_chart = df_chart.ffill().dropna()

            # Compute daily return relative to previous day
            # (Close_t / Close_{t-1}) - 1  -> decimal
            df_chart = df_chart.pct_change().dropna()
            # Convert to percentage values for display (no % sign on axis)
            df_chart = df_chart * 100
            metrics['chart_data'] = df_chart

            # --- Impact Strength Calculation ---
            # Define logic for classifying impact
            def classify_strength(return_val, vol_z_score):
                abs_ret = abs(return_val)
                if abs_ret > 2.0 or vol_z_score > 2.0:
                    return "Strong"
                elif abs_ret > 1.0:
                    return "Medium"
                else:
                    return "Weak"

            # Access T+0 Metrics for Strength Check
            t0_metrics = metrics.get('T+0', {})
            vol_z_t0 = t0_metrics.get('volume_z_score', 0.0)

            # 1. T+0 Impact (Initial Shock)
            metrics['impact_strength_t0'] = classify_strength(metrics['abnormal_return'], vol_z_t0)

            # 2. T+3 Impact (Sustained) - Only if we have more than 1 day of data
            metrics['impact_strength_t3'] = classify_strength(metrics['car_3d'], vol_z_t0)
            
            # 3. Composite/Default Impact (Dynamic)
            # If only T+0 is available (e.g., today's news), use T+0
            if (datetime.now().date() - target_ts).days < 3:
                 metrics['impact_strength'] = metrics['impact_strength_t0']
            else:
                 metrics['impact_strength'] = metrics['impact_strength_t3']
            
            # Sentiment check with Session-Awareness
            # Determine if news was After Market (T+1 reaction) or During/Pre Market (T+0 reaction)
            reaction_day = 'T+0' # Default
            
            if news_time_str:
                try:
                    import pytz
                    
                    # 1. Parse the Time
                    dt_parsed = pd.to_datetime(str(news_time_str))
                    
                    # 2. Anchor to the Event Date (date_obj) assuming input is US Eastern Time on that date
                    # date_obj is the US date of the news event.
                    t0_date = date_obj.date() if hasattr(date_obj, 'date') else date_obj
                    
                    # Combine Event Date + Parsed Time
                    dt_anchored = datetime.combine(t0_date, dt_parsed.time())
                    
                    eastern = pytz.timezone('America/New_York')
                    jst = pytz.timezone('Asia/Tokyo')
                    
                    # Localize as US EST
                    dt_est = eastern.localize(dt_anchored)
                    
                    # Convert to JST
                    dt_jst = dt_est.astimezone(jst)
                    
                    # Granular Logic with Date Check
                    news_date_jst = dt_jst.date()
                    day_diff = (news_date_jst - t0_date).days
                    h = dt_jst.hour
                    
                    if day_diff >= 1:
                        # News is on T+1 (or later) JST
                        reaction_day = 'T+1'
                        if h < 9:
                            timing_label = "Before T+1 Open"
                        elif h >= 15:
                             timing_label = "After T+1 Close"
                        else:
                            timing_label = "During T+1 Market Hours"
                            
                    elif day_diff < 0:
                         # News was T-1 JST (Rare, maybe previous night US?)
                         reaction_day = 'T+0'
                         timing_label = "Before T+0 Open" # Effectively
                         
                    else:
                        # Same Day (T+0)
                        if h >= 15:
                            reaction_day = 'T+1'
                            timing_label = "After T+0 Close"
                        elif h < 9:
                            reaction_day = 'T+0' 
                            timing_label = "Before T+0 Open"
                        else:
                            reaction_day = 'T+0'
                            timing_label = "During T+0 Market Hours"
                except:
                    pass
            
            # Store label
            metrics['timing_label'] = timing_label

            # Fallback Check: Does T+1 data exist?
            # If we determined T+1 but T+1 hasn't happened yet, fall back to T+0
            t1_exists = metrics.get('T+1', {}).get('close', 0) > 0
            
            if reaction_day == 'T+1' and not t1_exists:
                reaction_day = 'T+0'
                
            reaction_metrics = metrics.get(reaction_day, {})
            # Use Absolute Change for alignment (Intuitively: Good News -> Price Up)
            reaction_pct = reaction_metrics.get('pct_change_day', 0.0)

            metrics['sentiment_alignment'] = "N/A" # Default
            metrics['reaction_day_used'] = reaction_day 
            
            # Determine which return metric to use for alignment
            # Default to short-term reaction
            alignment_return = reaction_pct
            
            # If event is old enough to have T+3 data (approx 3 days), use CAR 3d
            # Use car_3d if it's non-zero or we are confident.
            # We use the time difference to decide preference.
            try:
                days_since = (datetime.now().date() - target_ts).days
                if days_since >= 3:
                    alignment_return = metrics.get('car_3d', 0.0)
            except:
                pass

            if sentiment_score is not None:
                try:
                    score = float(sentiment_score)
                    # Threshold 0.1
                    if (score > 0.1 and alignment_return > 0) or (score < -0.1 and alignment_return < 0):
                        metrics['sentiment_alignment'] = "Aligned"
                    elif (score > 0.1 and alignment_return < 0) or (score < -0.1 and alignment_return > 0):
                        metrics['sentiment_alignment'] = "Diverged"
                    else:
                        metrics['sentiment_alignment'] = "Neutral"
                except:
                    pass

        except Exception as e:
            print(f"Error fetching data: {e}")
            metrics = {
                'close': 0, 'pct_change_day': 0, 'volume_rel': 0, 
                'index_pct_change': 0, 'relative_change': 0, 
                'car_3d': 0, 'car_pre_3d': 0, 
                'impact_strength': 'N/A', 'impact_strength_t0': 'N/A', 'impact_strength_t3': 'N/A',
                'sentiment_alignment': 'N/A',
                'index_close': 0, 'intraday_change': 0, 'relative_intraday': 0, 'gap_pct': 0, 'sigma_move': 0,
                'chart_data': None,
                'date_event': None, 'date_t_minus_3': None, 'date_t_plus_3': None,
                'is_mock': True
            }

        self.market_data_cache[key] = metrics
        return metrics

    def save_feedback(self, feedback_data):
        """
        Appends feedback to the CSV log.
        feedback_data: dict
        """
        # Add timestamp
        feedback_data['timestamp'] = datetime.now().isoformat()
        
        df = pd.DataFrame([feedback_data])
        
        if not os.path.exists(self.feedback_path):
            df.to_csv(self.feedback_path, index=False)
        else:
            df.to_csv(self.feedback_path, mode='a', header=False, index=False)
            
    def get_feedback_stats(self):
        if not os.path.exists(self.feedback_path):
            return 0
        try:
            return len(pd.read_csv(self.feedback_path))
        except:
            return 0
