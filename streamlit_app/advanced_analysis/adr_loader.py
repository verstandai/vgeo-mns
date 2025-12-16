
import os
import json
import pandas as pd

def load_adr_data(base_dir=None):
    """
    Loops through all JSON files in the 'adr_json' folder and loads them into a Pandas DataFrame.
    """
    # If no base_dir provided, assume relative to this file
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    
    json_dir = os.path.join(base_dir, 'adr_json')
    
    if not os.path.exists(json_dir):
        print(f"Directory not found: {json_dir}")
        return pd.DataFrame()
    
    all_records = []
    files_processed = 0
    
    print(f"Reading JSON files from: {json_dir}")
    for filename in os.listdir(json_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(json_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Handle if the json file contains a list of records or a single record
                    if isinstance(data, list):
                        all_records.extend(data)
                    elif isinstance(data, dict):
                        all_records.append(data)
                    files_processed += 1
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                
    df = pd.DataFrame(all_records)
    print(f"Successfully processed {files_processed} CSV files.")
    print(f"Loaded {len(df)} total records.")
    return df

if __name__ == "__main__":
    # Test the loader
    df = load_adr_data()
    if not df.empty:
        print("\nDataFrame Info:")
        print(df.info())
        print("\nFirst 5 rows:")
        print(df.head())
