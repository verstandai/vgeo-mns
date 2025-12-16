
import pandas as pd
import os
import json
import ast

def load_adr_mapping(base_dir):
    """
    Loads ADR JSON files and creates a mapping dictionary centered on US Ticker.
    Returns a dict: { 'US_TICKER': { 'sector': ..., 'jp_peers': [...], 'us_peers': [...] } }
    """
    json_dir = os.path.join(base_dir, 'adr_json')
    mapping = {}
    
    if not os.path.exists(json_dir):
        print(f"Directory not found: {json_dir}")
        return mapping

    print(f"Loading ADR metadata from {json_dir}...")
    for filename in os.listdir(json_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(json_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # Normalize to list to handle both single dict and list of dicts
                    if isinstance(data, dict):
                        data = [data]
                        
                    for entry in data:
                        info = entry.get('adr_info', {})
                        us_ticker = info.get('US_ticker')
                        
                        if us_ticker:
                            # Extract Key Metadata
                            sector = info.get('sector_list', [])
                            sector_str = sector[0] if isinstance(sector, list) and len(sector) > 0 else "Unknown"
                            
                            # Extract Peers (List of Tickers)
                            jp_peers_dict = info.get('JP_peer_group', {})
                            us_peers_dict = info.get('US_peer_group', {})
                            
                            jp_peers = [v['US_ticker'] for k, v in jp_peers_dict.items() if 'US_ticker' in v]
                            us_peers = [v['US_ticker'] for k, v in us_peers_dict.items() if 'US_ticker' in v]
                            
                            mapping[us_ticker] = {
                                'primary_sector': sector_str,
                                'jp_peers': jp_peers,
                                'us_peers': us_peers,
                                'all_peers': jp_peers + us_peers,
                                'segments': list(info.get('business_segment_categories', {}).keys())
                            }
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                
    print(f"Mapped metadata for {len(mapping)} tickers.")
    return mapping

def enrich_news_with_metadata(aligned_csv_path, base_dir):
    """
    Reads the Aligned Records CSV and merges it with ADR metadata.
    """
    if not os.path.exists(aligned_csv_path):
        print(f"Aligned Data CSV not found at: {aligned_csv_path}")
        return None

    # Load News Data
    print(f"Loading news data from {aligned_csv_path}...")
    df_news = pd.read_csv(aligned_csv_path)
    
    # Load Metadata Mapping
    adr_map = load_adr_mapping(base_dir)
    
    # Prepare new columns
    sectors = []
    peer_data = []
    has_metadata = []
    
    for index, row in df_news.iterrows():
        ticker = row.get('us_ticker_name')
        
        # Clean ticker if needed (sometimes might be 'MUFG' or 'MUFG US')
        # Assuming simple match for now based on previous context
        meta = adr_map.get(ticker)
        
        if meta:
            sectors.append(meta['primary_sector'])
            peer_data.append(str(meta['all_peers'])) # Store as string representation of list
            has_metadata.append(True)
        else:
            sectors.append("Unknown")
            peer_data.append("[]")
            has_metadata.append(False)
            
    # Enrich DataFrame
    df_news['sector'] = sectors
    df_news['peers_list'] = peer_data
    df_news['has_adr_meta'] = has_metadata
    
    # Save enriched file
    output_path = os.path.join(base_dir, "aligned_records_with_peers.csv")
    df_news.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"✅ Enriched data saved to: {output_path}")
    print(f"Matches found: {sum(has_metadata)} out of {len(df_news)}")
    
    return df_news

if __name__ == "__main__":
    # Determine directories
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Path to your previously extracted Aligned CSV
    # Assuming it's in the same folder as this script
    aligned_csv = os.path.join(current_dir, "aligned_records_only.csv")
    
    # If aligned_recods_only.csv doesn't exist, try to generate it first using the existing tool logic?
    # For now, let's assume valid path or fallback to the main export validation
    if not os.path.exists(aligned_csv):
        print(f"Warning: {aligned_csv} not found. Using main export CSV for demo.")
        # Fallback to look for the main export in the parent directory
        parent_dir = os.path.dirname(current_dir)
        aligned_csv = os.path.join(parent_dir, "export_mns_demo_enriched.csv")
    
    enrich_news_with_metadata(aligned_csv, current_dir)
