
import pandas as pd
import yfinance as yf
import os
import ast
import time
from datetime import timedelta

class PeerContagionAnalyzer:
    def __init__(self, enriched_csv_path):
        self.csv_path = enriched_csv_path
        self.df = None
        self.peer_prices = {} # Cache: { 'TICKER_DATE': percent_change }

    def load_data(self):
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"File not found: {self.csv_path}")
        self.df = pd.read_csv(self.csv_path)
        # Ensure peers_list is a list, not a string
        self.df['peers_list'] = self.df['peers_list'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        print(f"Loaded {len(self.df)} records.")

    def fetch_peer_market_data(self):
        """
        Iterates through the dataset, identifies unique (Peer, Date) tuples,
        and fetches their market data from yfinance.
        """
        if self.df is None:
            self.load_data()

        # 1. Identify all unique requests needed
        # Structure: Set of (Ticker, DateStr)
        requests = set()
        
        for idx, row in self.df.iterrows():
            date_str = pd.to_datetime(row['date']).strftime('%Y-%m-%d')
            peers = row['peers_list']
            for peer in peers:
                requests.add((peer, date_str))
        
        print(f"Identified {len(requests)} unique peer price checks needed.")
        
        # 2. Fetch Data (Optimized by Ticker)
        # Group dates by ticker to minimize API calls
        ticker_dates = {}
        for ticker, date in requests:
            if ticker not in ticker_dates:
                ticker_dates[ticker] = []
            ticker_dates[ticker].append(date)
            
        print(f"Fetching data for {len(ticker_dates)} unique peers...")
        
        for ticker, dates in ticker_dates.items():
            # Find min and max date to pull a range (more efficient than 1-by-1)
            sorted_dates = sorted(dates)
            start_date = (pd.to_datetime(sorted_dates[0]) - timedelta(days=5)).strftime('%Y-%m-%d')
            end_date = (pd.to_datetime(sorted_dates[-1]) + timedelta(days=5)).strftime('%Y-%m-%d')
            
            try:
                # Download Ticker Data
                # Auto-adjust: If Ticker is strict US (e.g. JPM), just use JPM.
                # If these are JP peers in US format (e.g. 8306 JP -> 8306.T is complicated),
                # We assume the peers_list contains US Tickers (e.g. HMC for Honda, TM for Toyota).
                # Based on previous JSON, they seem to be matching US tickers like 'HMC', 'TM'.
                
                print(f"Processing {ticker}...")
                data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                
                if not data.empty:
                    # Calculate Daily Pct Change (Close-to-Close for simplicity)
                    # For T+1 analysis, we might ideally want Open-Close, but let's stick to standard daily % first.
                    data['PctChange'] = data['Close'].pct_change() * 100
                    
                    # Store relevant dates in cache
                    for d in dates:
                        # Find the closest trading day if exact match fails?
                        # Using exact match for now.
                        lookup_d = pd.to_datetime(d)
                        if lookup_d in data.index:
                            val = data.loc[lookup_d]['PctChange']
                             # Handle scalar vs Series (yf sometimes returns Series for one row)
                            val = val.item() if hasattr(val, 'item') else val
                            self.peer_prices[(ticker, d)] = val
                        else:
                            # Try T+1 (Next Day)
                            lookup_d_next = lookup_d + timedelta(days=1)
                            if lookup_d_next in data.index:
                                val = data.loc[lookup_d_next]['PctChange']
                                val = val.item() if hasattr(val, 'item') else val
                                self.peer_prices[(ticker, d)] = val # Map original date request to next day's price
            
            except Exception as e:
                print(f"Failed to fetch {ticker}: {e}")
            
            # Rate limit politeness
            time.sleep(0.1)

    def analyze_contagion(self):
        """
        Calculates correlation stats.
        """
        results = []
        
        for idx, row in self.df.iterrows():
            target_ticker = row['us_ticker_name']
            date_str = pd.to_datetime(row['date']).strftime('%Y-%m-%d')
            target_move = row['market_t1_stock_pct_change'] # Using T+1 as the "Reaction"
            peers = row['peers_list']
            sector = row['sector']
            
            peer_moves = []
            same_direction_count = 0
            
            for peer in peers:
                peer_move = self.peer_prices.get((peer, date_str))
                if peer_move is not None and not pd.isna(peer_move):
                    peer_moves.append(peer_move)
                    # Check Direction Match
                    if (target_move > 0 and peer_move > 0) or (target_move < 0 and peer_move < 0):
                        same_direction_count += 1
            
            if peer_moves:
                avg_peer_move = sum(peer_moves) / len(peer_moves)
                contagion_score = same_direction_count / len(peer_moves) # % of peers moving in same direction
                
                results.append({
                    'Target': target_ticker,
                    'Date': date_str,
                    'Sector': sector,
                    'Sentiment': row['news_sentiment'],
                    'Target_Move': target_move,
                    'Avg_Peer_Move': avg_peer_move,
                    'Contagion_Ratio': contagion_score,
                    'Num_Peers': len(peer_moves)
                })
                
        # Convert to DataFrame
        df_res = pd.DataFrame(results)
        
        # Save
        if not df_res.empty:
            output_file = "peer_contagion_analysis.csv"
            df_res.to_csv(os.path.join(os.path.dirname(self.csv_path), output_file), index=False)
            print(f"Contagion Analysis saved to: {output_file}")
            
            # Print Summary Stats
            print("\n--- Contagion Summary ---")
            print(f"Average Contagion Ratio (Sync): {df_res['Contagion_Ratio'].mean():.2%}")
            print(f"Correlation (Target Move vs Avg Peer Move): {df_res['Target_Move'].corr(df_res['Avg_Peer_Move']):.2f}")
            
            try:
                print("\nTop Sectors by Sympathy:")
                print(df_res.groupby('Sector')['Contagion_Ratio'].mean().sort_values(ascending=False))
            except:
                pass

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "aligned_records_with_peers.csv")
    
    analyzer = PeerContagionAnalyzer(csv_path)
    analyzer.fetch_peer_market_data()
    analyzer.analyze_contagion()
