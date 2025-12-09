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
        
        df = pd.read_csv(self.data_path)
        
        # Parse Dates
        # Try multiple formats or use coerce to handle errors
        df['parsed_date'] = pd.to_datetime(df['date'], format='%m/%d/%y', errors='coerce').dt.date
        
        # Fallback for other formats if needed (though the demo data seems consistent)
        mask = df['parsed_date'].isna()
        if mask.any():
            df.loc[mask, 'parsed_date'] = pd.to_datetime(df.loc[mask, 'date'], errors='coerce').dt.date

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

    def get_market_data(self, ticker, date_obj, local_ticker=None, index_name=None, sentiment_score=None):
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
            
            # Define window: Start = Date - 40 days (for volatility/avg), End = Date + 10
            # Convert to string to ensure yfinance handles it correctly
            start_date = (date_obj - timedelta(days=40)).strftime('%Y-%m-%d')
            end_date = (date_obj + timedelta(days=10)).strftime('%Y-%m-%d')
            
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

            # Get Data for Target Day
            day_data = hist.loc[target_ts]
            
            # Get Previous Trading Day
            # Use integer location to be safe
            try:
                target_idx = hist.index.get_loc(target_ts)
                if target_idx > 0:
                    prev_close = hist.iloc[target_idx - 1]['Close']
                else:
                    # Fallback if it's the first day in the fetched history
                    prev_close = day_data['Open']
            except Exception:
                prev_close = day_data['Open']

            # --- Basic Metrics ---
            metrics['close'] = day_data['Close']
            metrics['open'] = day_data['Open']
            metrics['pct_change_day'] = ((day_data['Close'] - prev_close) / prev_close) * 100
            metrics['volume'] = day_data['Volume']
            
            # Gap % (Overnight Move: Open vs Prev Close)
            metrics['gap_pct'] = ((day_data['Open'] - prev_close) / prev_close) * 100
            
            # Intraday Trend (Close vs Open)
            metrics['intraday_change'] = ((day_data['Close'] - day_data['Open']) / day_data['Open']) * 100
            
            # Sigma Move (Z-Score)
            # Use past 30 trading days for volatility
            past_hist = hist.loc[:target_ts].iloc[:-1] # Exclude today
            if len(past_hist) >= 20:
                # Calculate daily returns
                past_returns = past_hist['Close'].pct_change().dropna()
                # Take last 30 days
                window_returns = past_returns.tail(30)
                
                mean_ret = window_returns.mean()
                std_ret = window_returns.std()
                
                day_ret = metrics['pct_change_day'] / 100.0
                
                if std_ret > 0:
                    metrics['sigma_move'] = (day_ret - mean_ret) / std_ret
                else:
                    metrics['sigma_move'] = 0.0
            else:
                metrics['sigma_move'] = 0.0
            
            # Avg Volume & Std Dev (past 20 days)
            window_hist = hist.loc[:target_ts].iloc[-21:-1] # Last 20 days before today
            if not window_hist.empty:
                avg_vol = window_hist['Volume'].mean()
                std_vol = window_hist['Volume'].std()
                metrics['volume_rel'] = metrics['volume'] / avg_vol if avg_vol else 1.0
                metrics['volume_z_score'] = (metrics['volume'] - avg_vol) / std_vol if std_vol else 0.0
            else:
                metrics['volume_rel'] = 1.0
                metrics['volume_z_score'] = 0.0

            # --- Index Metrics ---
            metrics['index_pct_change'] = 0.0
            metrics['index_close'] = 0.0
            metrics['index_intraday'] = 0.0
            
            if not hist_idx.empty and target_ts in hist_idx.index:
                idx_day = hist_idx.loc[target_ts]
                metrics['index_close'] = idx_day['Close']
                metrics['index_intraday'] = ((idx_day['Close'] - idx_day['Open']) / idx_day['Open']) * 100
                
                past_idx = hist_idx.loc[:target_ts].iloc[:-1]
                if not past_idx.empty:
                    prev_idx_close = past_idx.iloc[-1]['Close']
                    metrics['index_pct_change'] = ((idx_day['Close'] - prev_idx_close) / prev_idx_close) * 100
            
            metrics['relative_change'] = metrics['pct_change_day'] - metrics['index_pct_change']
            metrics['relative_intraday'] = metrics['intraday_change'] - metrics['index_intraday']
            
            # --- CAR Calculation (T+0 to T+3) ---
            car = 0.0
            # Get dates from target_ts onwards
            post_dates = [d for d in hist.index if d >= target_ts][:4] # T+0 to T+3
            
            car_t0 = 0.0 # T+0 only
            
            for i, d in enumerate(post_dates):
                # Stock Return
                d_data = hist.loc[d]
                d_prev = hist.loc[:d].iloc[-2]['Close'] if hist.loc[:d].shape[0] >= 2 else d_data['Open']
                s_ret = (d_data['Close'] - d_prev) / d_prev
                
                # Index Return
                i_ret = 0.0
                if not hist_idx.empty and d in hist_idx.index:
                    i_data = hist_idx.loc[d]
                    i_prev = hist_idx.loc[:d].iloc[-2]['Close'] if hist_idx.loc[:d].shape[0] >= 2 else i_data['Open']
                    i_ret = (i_data['Close'] - i_prev) / i_prev
                
                daily_ar = (s_ret - i_ret)
                car += daily_ar
                if i == 0:
                    car_t0 = daily_ar
            
            metrics['car_3d'] = car * 100 # Convert to percentage
            metrics['abnormal_return'] = car_t0 * 100
            
            # --- Pre-Event CAR Calculation (T-3 to T-1) ---
            car_pre = 0.0
            # Get dates strictly before target_ts
            pre_dates = [d for d in hist.index if d < target_ts][-3:] # Last 3 days before event
            
            for d in pre_dates:
                # Stock Return
                d_data = hist.loc[d]
                d_prev = hist.loc[:d].iloc[-2]['Close'] if hist.loc[:d].shape[0] >= 2 else d_data['Open']
                s_ret = (d_data['Close'] - d_prev) / d_prev
                
                # Index Return
                i_ret = 0.0
                if not hist_idx.empty and d in hist_idx.index:
                    i_data = hist_idx.loc[d]
                    i_prev = hist_idx.loc[:d].iloc[-2]['Close'] if hist_idx.loc[:d].shape[0] >= 2 else i_data['Open']
                    i_ret = (i_data['Close'] - i_prev) / i_prev
                
                car_pre += (s_ret - i_ret)
            
            metrics['car_pre_3d'] = car_pre * 100
            
            # Store Key Dates for Charting
            metrics['date_event'] = target_ts
            metrics['date_t_minus_3'] = pre_dates[0] if pre_dates else None
            metrics['date_t_plus_3'] = post_dates[-1] if len(post_dates) > 3 else (post_dates[-1] if post_dates else None)

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

            # Strong: CAR > 2% OR Volume Z > 2.0
            # Medium: CAR > 1%
            # Weak: CAR <= 1%
            abs_car = abs(metrics['car_3d'])
            if abs_car > 2.0 or metrics['volume_z_score'] > 2.0:
                metrics['impact_strength'] = "Strong"
            elif abs_car > 1.0:
                metrics['impact_strength'] = "Medium"
            else:
                metrics['impact_strength'] = "Weak"
                
            # --- Sentiment Alignment ---
            metrics['sentiment_alignment'] = "N/A"
            if sentiment_score is not None:
                try:
                    score = float(sentiment_score)
                    # Aligned if signs match (and magnitude is relevant)
                    if (score > 0.2 and metrics['car_3d'] > 0) or (score < -0.2 and metrics['car_3d'] < 0):
                        metrics['sentiment_alignment'] = "Aligned"
                    elif (score > 0.2 and metrics['car_3d'] < -0.5) or (score < -0.2 and metrics['car_3d'] > 0.5):
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
                'car_3d': 0, 'car_pre_3d': 0, 'impact_strength': 'N/A', 'sentiment_alignment': 'N/A',
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
