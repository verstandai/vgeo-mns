import pandas as pd
import os

def check_csv():
    # Attempt to locate the CSV file
    possible_files = [
        '../mns_demo_enriched.csv', 
        '../mns_demo_enriched_2.csv',
        'mns_demo_enriched.csv'
    ]
    
    file_path = None
    for f in possible_files:
        if os.path.exists(f):
            file_path = f
            print(f"Found file at: {f}")
            break
            
    if not file_path:
        print("❌ No CSV file found in expected locations.")
        return

    # Load Data
    try:
        df = pd.read_csv(file_path)
    except:
        df = pd.read_csv(file_path, encoding='utf-8-sig')

    print(f"✅ Loaded {len(df)} rows.")
    
    # List Columns
    print("\n--- Columns ---")
    for c in sorted(df.columns):
        print(f"• {c}")

    # Check for Critical App Columns
    print("\n--- Critical Column Check ---")
    critical_cols = [
        'company_name',
        'local_ticker',
        'index',
        'news_sentiment',    
        'breaking_recap',        
        'classification',        
        'date',
        'timestamp',
        'actionable_intelligence',
        'event_category',
        'headline',
        'job_load_timestamp',
        'key_takeaways',
        'llm_model',
        'main_content',
        'main_content_en',
        'news_sentiment',
        'news_worthy_key_factors',
        'news_worthy_reasoning',
        'original_headline',
        'original_language',
        'pipeline_executed_timestamp',
        'reporter',
        'sentiment',
        'sentiment_reasoning',
        'source',
        'timestamp',
        'translated_language',
        'url',
        'us_ticker_name'
    ]
    
    for c in critical_cols:
        if c in df.columns:
            print(f"✅ {c}")
        else:
            print(f"❌ {c} (Missing)")

if __name__ == "__main__":
    check_csv()