
import pandas as pd
import os

def print_stats(df_subset, subset_name):
    print(f"\n{'='*20} {subset_name} (n={len(df_subset)}) {'='*20}")
    
    if len(df_subset) == 0:
        print("No records found.")
        return

    neutral_threshold = 0.05
    
    # ---------------------------------------------------------
    # 1. T+1 Immediate Reaction (Stock % Change)
    # ---------------------------------------------------------
    print("[ T+1 (Next Day) Stock Price Change ]")
    
    pos_sent = df_subset[df_subset['news_sentiment'] > neutral_threshold]
    if len(pos_sent) > 0:
        pos_correct = pos_sent[pos_sent['market_t1_stock_pct_change'] > 0]
        pos_acc = len(pos_correct) / len(pos_sent) * 100
    else:
        pos_correct = []
        pos_acc = 0.0
    
    neg_sent = df_subset[df_subset['news_sentiment'] < -neutral_threshold]
    if len(neg_sent) > 0:
        neg_correct = neg_sent[neg_sent['market_t1_stock_pct_change'] < 0]
        neg_acc = len(neg_correct) / len(neg_sent) * 100
    else:
        neg_correct = []
        neg_acc = 0.0
    
    print(f"Positive Sentiment Count: {len(pos_sent)}")
    print(f"  -> Price Up: {len(pos_correct)} ({pos_acc:.1f}%)")
    
    print(f"Negative Sentiment Count: {len(neg_sent)}")
    print(f"  -> Price Down: {len(neg_correct)} ({neg_acc:.1f}%)")
    
    total_signals = len(pos_sent) + len(neg_sent)
    total_correct = len(pos_correct) + len(neg_correct)
    overall_acc = total_correct / total_signals * 100 if total_signals > 0 else 0
    print(f"Overall Directional Accuracy: {overall_acc:.1f}%")

    # ---------------------------------------------------------
    # 2. T+3 Sustained Impact (CAR 3-Day)
    # ---------------------------------------------------------
    print("\n[ T+3 Cumulative Abnormal Return (CAR) ]")
    
    if len(pos_sent) > 0:
        pos_correct_car = pos_sent[pos_sent['market_car_3d'] > 0]
        pos_acc_car = len(pos_correct_car) / len(pos_sent) * 100
    else:
        pos_correct_car = []
        pos_acc_car = 0
    
    if len(neg_sent) > 0:
        neg_correct_car = neg_sent[neg_sent['market_car_3d'] < 0]
        neg_acc_car = len(neg_correct_car) / len(neg_sent) * 100
    else:
        neg_correct_car = []
        neg_acc_car = 0
    
    print(f"Positive Sentiment -> T+3 CAR > 0: {pos_acc_car:.1f}%")
    print(f"Negative Sentiment -> T+3 CAR < 0: {neg_acc_car:.1f}%")
    
    total_correct_car = len(pos_correct_car) + len(neg_correct_car)
    overall_acc_car = total_correct_car / total_signals * 100 if total_signals > 0 else 0
    print(f"Overall T+3 Directional Accuracy: {overall_acc_car:.1f}%")

    # ---------------------------------------------------------
    # 3. Relative Alpha (T+1 Relative to Index)
    # ---------------------------------------------------------
    if 'market_t1_relative_index_change' in df_subset.columns:
        # Filter for rows that actually have this metric (though passed subset might have NaNs)
        df_rel = df_subset.dropna(subset=['market_t1_relative_index_change'])
        
        print("\n[ T+1 Relative to Index (Alpha) ]")
        
        pos_sent_rel = df_rel[df_rel['news_sentiment'] > neutral_threshold]
        neg_sent_rel = df_rel[df_rel['news_sentiment'] < -neutral_threshold]
        
        pos_correct_rel = []
        neg_correct_rel = []

        if len(pos_sent_rel) > 0:
            pos_correct_rel = pos_sent_rel[pos_sent_rel['market_t1_relative_index_change'] > 0]
            pos_acc_rel = len(pos_correct_rel) / len(pos_sent_rel) * 100
        else:
            pos_acc_rel = 0
            
        if len(neg_sent_rel) > 0:
            neg_correct_rel = neg_sent_rel[neg_sent_rel['market_t1_relative_index_change'] < 0]
            neg_acc_rel = len(neg_correct_rel) / len(neg_sent_rel) * 100
        else:
            neg_acc_rel = 0
        
        print(f"Positive Sentiment -> Outperformed Index: {pos_acc_rel:.1f}%")
        print(f"Negative Sentiment -> Underperformed Index: {neg_acc_rel:.1f}%")
        
        total_signals_rel = len(pos_sent_rel) + len(neg_sent_rel)
        total_correct_rel = len(pos_correct_rel) + len(neg_correct_rel)
        overall_acc_rel = total_correct_rel / total_signals_rel * 100 if total_signals_rel > 0 else 0
        print(f"Overall Relative Accuracy: {overall_acc_rel:.1f}%")


def analyze_predictions():
    # Load the exported data
    file_path = "../export_mns_demo_enriched.csv"
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    df = pd.read_csv(file_path)

    # Filter for rows with necessary data
    # Drop rows where critical metrics are NaN
    df_clean = df.dropna(subset=['news_sentiment', 'market_t1_stock_pct_change', 'market_car_3d'])
    
    # 1. All News
    print_stats(df_clean, "ALL NEWS")
    
    # 2. Significant News
    df_sig = df_clean[df_clean['classification'] == 'SIGNIFICANT']
    print_stats(df_sig, "SIGNIFICANT NEWS")
    
    # 3. Insignificant News
    df_insig = df_clean[df_clean['classification'] == 'INSIGNIFICANT']
    print_stats(df_insig, "INSIGNIFICANT NEWS")

    # 4. Breaking News
    df_breaking = df_clean[df_clean['breaking_recap'] == 'breaking']
    print_stats(df_breaking, "BREAKING NEWS")

    # 5. Recap News
    df_recap = df_clean[df_clean['breaking_recap'] == 'recap']
    print_stats(df_recap, "RECAP NEWS")

    # 6. Group by Source
    unique_sources = df_clean['source_en'].unique()
    for source in unique_sources:
        # Clean source name for display
        if pd.isna(source):
            continue
        df_source = df_clean[df_clean['source_en'] == source]
        print_stats(df_source, f"SOURCE: {source}")

    # 7. Group by Event Category
    unique_categories = df_clean['event_category'].unique()
    for category in unique_categories:
        if pd.isna(category):
            continue
        df_cat = df_clean[df_clean['event_category'] == category]
        print_stats(df_cat, f"CATEGORY: {category}")

    # 8. Golden Strategy
    # Criteria:
    # - Event: Partnership OR Earnings Report
    # - Source: gamebiz OR MINKABU
    # - Type: Breaking
    # - Class: Significant
    df_golden = df_clean[
        (df_clean['event_category'].isin(['Partnership', 'Earnings Report'])) &
        (df_clean['source_en'].isin(['gamebiz', 'MINKABU'])) &
        (df_clean['breaking_recap'] == 'breaking') &
        (df_clean['classification'] == 'SIGNIFICANT')
    ]
    print_stats(df_golden, "GOLDEN STRATEGY (Combined Filters)")

if __name__ == "__main__":
    analyze_predictions()
