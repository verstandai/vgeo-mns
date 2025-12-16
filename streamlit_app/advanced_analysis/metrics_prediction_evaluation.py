
import pandas as pd
import numpy as np
import os
from sklearn.tree import DecisionTreeClassifier, export_text

class MultivariateAnalyzer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None

    def load_data(self):
        """Loads data from the CSV file."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found at: {self.data_path}")
        
        self.df = pd.read_csv(self.data_path)
        print(f"Loaded {len(self.df)} records.")
        return self.df

    def engineer_features(self):
        """
        Phase 1: Feature Engineering
        Creates interaction terms, bins continuous variables, and adds specific factors.
        """
        if self.df is None:
            self.load_data()
        
        df = self.df.copy()

        # 1. Normalize/Fill NaNs for Safety
        # Replacing infinite values with NaNs then dropping or filling is safer
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Ensure numeric columns are actually numeric
        numeric_cols = [
            'news_sentiment', 'market_t0_volume_ratio', 'market_t0_sigma_move',
            'market_t1_gap_pct', 'market_t1_stock_intraday', 'market_t1_stock_pct_change'
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # 2. Interaction Terms
        # Sentiment-Volume Score: Sentiment * Volume Ratio
        # (High Confidence + High Volume = Strong Signal)
        if 'news_sentiment' in df.columns and 'market_t0_volume_ratio' in df.columns:
            df['factor_sent_vol'] = df['news_sentiment'] * df['market_t0_volume_ratio'].fillna(1.0)

        # Volatility-Adjusted Sentiment: Sentiment / Sigma
        # (Is the sentiment "strong" relative to the price noise?)
        if 'news_sentiment' in df.columns and 'market_t0_sigma_move' in df.columns:
            # Avoid division by zero
            df['factor_sent_sigma'] = df.apply(
                lambda row: row['news_sentiment'] / row['market_t0_sigma_move'] 
                if pd.notnull(row['market_t0_sigma_move']) and abs(row['market_t0_sigma_move']) > 0.01 
                else 0, axis=1
            )

        # 3. Discretize Variables (Binning)
        # Volume Buckets
        if 'market_t0_volume_ratio' in df.columns:
            df['bin_volume'] = pd.cut(
                df['market_t0_volume_ratio'],
                bins=[-np.inf, 0.8, 1.5, 3.0, np.inf],
                labels=['Low', 'Normal', 'High', 'Extreme']
            )

        # Gap Buckets
        if 'market_t1_gap_pct' in df.columns:
            df['bin_gap'] = pd.cut(
                df['market_t1_gap_pct'],
                bins=[-np.inf, -1.0, 1.0, np.inf],
                labels=['Gap Down', 'Flat', 'Gap Up']
            )

        # 4. Filter for Valid Analysis Set
        # We need T+1 outcomes to be present to test anything
        df_clean = df.dropna(subset=['market_t1_stock_pct_change', 'market_car_3d'])
        
        self.df = df_clean
        print(f"Feature Engineering Complete. Active Records: {len(self.df)}")
        return self.df

    def analyze_factors(self):
        """
        Phase 2: Simple Factor Analysis
        Groups by bins and calculates Win Rates.
        """
        # Example: Win Rate by Volume Bin
        print("\n--- Win Rate by Volume Bin (T+1) ---")
        if 'bin_volume' in self.df.columns:
            stats = self.df.groupby('bin_volume', observed=True)['market_t1_stock_pct_change'].apply(lambda x: (x > 0).mean())
            print(stats)

        # Example: Win Rate by Gap Bin
        print("\n--- Win Rate by Gap Bin (T+1) ---")
        if 'bin_gap' in self.df.columns:
            stats = self.df.groupby('bin_gap', observed=True)['market_t1_stock_pct_change'].apply(lambda x: (x > 0).mean())
            print(stats)
            
    def find_rules_decision_tree(self):
        """
        Trains a simple Decision Tree to find predictive rules for T+1 Direction.
        """
        print("\n--- Decision Tree Rule Extraction ---")
        # Prepare Data
        # Features: news_sentiment, volume_ratio, gap_pct, sigma_move
        features = ['news_sentiment', 'market_t0_volume_ratio', 'market_t1_gap_pct', 'market_t0_sigma_move']
        target = 'target_up'
        
        # Make a copy and drop rows where features are NaNs
        df_model = self.df.dropna(subset=features).copy()
        
        if len(df_model) == 0:
            print("Not enough data for Decision Tree.")
            return

        df_model['target_up'] = (df_model['market_t1_stock_pct_change'] > 0).astype(int)
        
        X = df_model[features]
        y = df_model[target]
        
        # Train simple tree (max_depth=3 for readability)
        clf = DecisionTreeClassifier(max_depth=3, random_state=42)
        clf.fit(X, y)
        
        # Print Rules
        r = export_text(clf, feature_names=features)
        print(r)
        
        # Feature Importance
        print("\nFeature Importances:")
        for name, imp in zip(features, clf.feature_importances_):
            print(f"{name}: {imp:.4f}")

    def simulate_gap_strategy(self):
        """
        Simulates the 'Gap Momentum' strategy found in preliminary analysis.
        Allocates capital equally to all valid signals.
        REALISTIC MODE: Uses 'market_t1_stock_intraday' (Open to Close return)
        to simulate entering AFTER the gap.
        FILTERED: Adds Sentiment Confirmation.
        """
        print("\n--- Backtest: Sentiment-Aligned Gap Momentum Strategy ---")
        # Rule: 
        # Long if Gap > 1.0% AND Sentiment > 0.2
        # Short if Gap < -1.0% AND Sentiment < -0.2
        
        initial_capital = 100000.0
        capital = initial_capital
        
        trades_count = 0
        winning_trades = 0
        total_pnl = 0.0
        
        # Fixed bet size per trade ($10,000)
        bet_size = 10000.0 
        trade_pnls_pct = []
        
        for index, row in self.df.iterrows():
            gap = row['market_t1_gap_pct']
            # Use Intraday Change (Open -> Close) instead of full day change
            ret_pct = row['market_t1_stock_intraday']
            # Get Sentiment
            sentiment = row['news_sentiment']
            
            if pd.isna(gap) or pd.isna(ret_pct) or pd.isna(sentiment):
                continue
                
            ret_decimal = ret_pct / 100.0
            position = 0
            
            # Sentiment-Aligned Strategy Logic
            if gap > 1.0 and sentiment > 0.2:
                position = 1 # Long
            elif gap < -1.0 and sentiment < -0.2:
                position = -1 # Short
            
            if position != 0:
                trades_count += 1
                pnl = bet_size * position * ret_decimal
                total_pnl += pnl
                capital += pnl
                
                # Record percentage return of this trade relative to bet size
                # (e.g. if PnL is $500 on $10k bet, return is 5%)
                trade_return = (pnl / bet_size)
                trade_pnls_pct.append(trade_return)
                
                if pnl > 0:
                    winning_trades += 1
        
        win_rate = winning_trades / trades_count if trades_count > 0 else 0
        roi = (total_pnl / initial_capital) * 100
        
        # Sharpe Calculation
        if len(trade_pnls_pct) > 1:
            avg_ret = np.mean(trade_pnls_pct)
            std_ret = np.std(trade_pnls_pct, ddof=1)
            
            # Annualization factor:
            # Assuming these are daily holding period trades.
            # If we trade roughly every day, factor is sqrt(252).
            # If these are sporadic, this is "Sharpe per Trade".
            # For standard comparison, let's show per-trade stats and annualized estimate.
            sharpe_per_trade = avg_ret / std_ret if std_ret > 0 else 0
            
            # To estimate annualized sharpe, we multiply by sqrt(Number of Trades per Year).
            # We don't strictly know the timespan, but let's assume the dataset covers roughly 2-3 weeks 
            # based on "December" dates seen. Let's stick to Per-Trade Sharpe for accuracy.
        else:
            sharpe_per_trade = 0
            std_ret = 0

        print(f"Strategy: Long if Gap > 1%, Short if Gap < -1%")
        print(f"Assumptions: ${bet_size:,.0f} per trade, no compounding.")
        print(f"Total Trades: {trades_count}")
        print(f"Win Rate: {win_rate:.1%}")
        print(f"Mean Return per Trade: {np.mean(trade_pnls_pct):.2%}")
        print(f"Std Dev per Trade: {std_ret:.2%}")
        print(f"Sharpe Ratio (Per Trade): {sharpe_per_trade:.2f}")
        print(f"  (Typically > 0.1 per trade is good, > 0.5 is excellent)")
        print(f"Total PnL: ${total_pnl:,.2f}")
        print(f"ROI: {roi:.1f}%")

if __name__ == "__main__":
    # Path assumption: Running from streamlit_app/advanced_analysis or similar
    # Adjust path to find the CSV in the parent directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, "export_mns_demo_enriched.csv")
    
    analyzer = MultivariateAnalyzer(csv_path)
    analyzer.engineer_features()
    analyzer.analyze_factors()
    analyzer.find_rules_decision_tree()
    analyzer.simulate_gap_strategy()
