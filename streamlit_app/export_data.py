import os
import pandas as pd
from data_manager import DataManager

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(os.path.dirname(BASE_DIR), "mns_demo_enriched.csv")
FEEDBACK_PATH = os.path.join(BASE_DIR, "feedback_log.csv")
EXPORT_PATH = os.path.join(BASE_DIR, "export_mns_demo_enriched.csv")

def main():
    print("Initializing DataManager...")
    manager = DataManager(DATA_PATH, FEEDBACK_PATH)
    
    print(f"Loading data from {DATA_PATH}...")
    df = manager.load_data()
    
    if df.empty:
        print("No data found to export.")
        return

    print(f"Enriching and exporting {len(df)} records...")
    
    # 1. Emulate App Logic: Add Sentiment Category
    def categorize_sentiment(score):
        try:
            s = float(score)
            if s > 0.25: return "Positive"
            if s < -0.25: return "Negative"
            return "Neutral"
        except:
            return "Neutral"
    df['sentiment_category'] = df['news_sentiment'].apply(categorize_sentiment)

    # 2. Emulate App Logic: Add Alignment
    print("Calculating alignment (this fetches market data)...")
    def get_alignment_status(row):
        # We fetch market data here just to get the alignment string
        # The export_market_data function will fetch it AGAIN unless cached.
        # Since DataManager has a cache, this is fine.
        md = manager.get_market_data(
            row['us_ticker_name'], 
            row['parsed_date'],
            local_ticker=row.get('local_ticker'),
            index_name=row.get('index'),
            sentiment_score=row.get('news_sentiment'),
            news_time_str=row.get('timestamp')
        )
        return md.get('sentiment_alignment', 'Unknown') if md else 'Unknown'

    df['alignment'] = df.apply(get_alignment_status, axis=1)

    try:
        # We use the manager's export, passing our pre-enriched df
        output_file = manager.export_market_data(df, EXPORT_PATH)
        print(f"✅ Successfully exported enriched data to: {output_file}")
    except Exception as e:
        print(f"❌ Export failed: {e}")

if __name__ == "__main__":
    main()
