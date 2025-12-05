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

    def get_market_data(self, ticker, date_obj, local_ticker=None, index_name=None):
        """
        Fetches market data for a specific ticker and date.
        Returns a dictionary of metrics.
        """
        if date_obj is None or pd.isna(date_obj):
            return None

        # Create a cache key
        key = f"{ticker}_{local_ticker}_{date_obj}"
        if key in self.market_data_cache:
            return self.market_data_cache[key]

        metrics = {}
        
        try:
            # Determine Ticker Symbol
            # If local_ticker is provided and looks like a Japanese code (4 digits), use it
            target_ticker = ticker
            if local_ticker and str(local_ticker).isdigit():
                target_ticker = f"{local_ticker}.T"
            
            # Determine Index Ticker
            index_ticker = "^TOPX" # Default
            if index_name == "TOPIX":
                index_ticker = "1306.T" # Use ETF as proxy if ^TOPX fails (which it does often)
            
            # Define window: Start = Date - 10 days (to get PrevClose), End = Date + 5
            start_date = date_obj - timedelta(days=10)
            end_date = date_obj + timedelta(days=5)
            
            # Fetch Stock Data
            stock = yf.Ticker(target_ticker)
            hist = stock.history(start=start_date, end=end_date)
            
            # Fetch Index Data
            idx = yf.Ticker(index_ticker)
            hist_idx = idx.history(start=start_date, end=end_date)

            if hist.empty:
                raise ValueError(f"No data found for {target_ticker}")

            # Normalize index to date only
            hist.index = hist.index.date
            if not hist_idx.empty:
                hist_idx.index = hist_idx.index.date
            
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
            
            # Get Previous Trading Day (for Day Change)
            # We need the row strictly before target_ts
            past_days = hist.loc[:target_ts].iloc[:-1]
            if past_days.empty:
                # Can't calc change without prev day
                prev_close = day_data['Open'] # Fallback
            else:
                prev_close = past_days.iloc[-1]['Close']

            # --- Stock Metrics ---
            metrics['close'] = day_data['Close']
            metrics['open'] = day_data['Open']
            
            # 1. Stock % Change on Day (Close vs Prev Close)
            metrics['pct_change_day'] = ((day_data['Close'] - prev_close) / prev_close) * 100
            
            # 2. Stock % Intraday Change (Close vs Open)
            metrics['intraday_change'] = ((day_data['Close'] - day_data['Open']) / day_data['Open']) * 100
            
            # 3. Stock % Change at Open (Open vs Prev Close)
            metrics['open_change_pct'] = ((day_data['Open'] - prev_close) / prev_close) * 100
            
            metrics['volume'] = day_data['Volume']
            
            # Avg Volume (past 5 days available in window)
            metrics['avg_volume'] = hist['Volume'].mean()
            metrics['volume_rel'] = metrics['volume'] / metrics['avg_volume'] if metrics['avg_volume'] else 1.0

            # --- Index Metrics ---
            metrics['index_pct_change'] = 0.0
            if not hist_idx.empty and target_ts in hist_idx.index:
                idx_day = hist_idx.loc[target_ts]
                
                # Find prev index day
                past_idx = hist_idx.loc[:target_ts].iloc[:-1]
                if not past_idx.empty:
                    prev_idx_close = past_idx.iloc[-1]['Close']
                    metrics['index_pct_change'] = ((idx_day['Close'] - prev_idx_close) / prev_idx_close) * 100
            
            # 4. Relative Change (Stock Day Chg - Index Day Chg)
            metrics['relative_change'] = metrics['pct_change_day'] - metrics['index_pct_change']
            
            # Abnormal Return (Simplified as Relative Change for now)
            metrics['abnormal_return'] = metrics['relative_change']

        except Exception as e:
            print(f"Error fetching data: {e}")
            # Fallback to mock if absolutely necessary, but try to avoid
            metrics = {
                'close': 0, 'pct_change_day': 0, 'intraday_change': 0,
                'open_change_pct': 0, 'relative_change': 0, 'abnormal_return': 0,
                'volume_rel': 0, 'is_mock': True
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
