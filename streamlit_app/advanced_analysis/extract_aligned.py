
import pandas as pd
import os

class AlignmentExtractor:
    def __init__(self, data_path, output_dir=None):
        self.data_path = data_path
        self.output_dir = output_dir if output_dir else os.path.dirname(data_path)
        self.df = None

    def load_data(self):
        """Loads data from the CSV file."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found at: {self.data_path}")
        
        self.df = pd.read_csv(self.data_path)
        print(f"Loaded {len(self.df)} records.")
        return self.df

    def extract_aligned_records(self):
        """
        Extracts records where 'alignment' is 'Aligned'.
        Saves the filtered dataset to a new CSV.
        """
        if self.df is None:
            self.load_data()
        
        # Check if 'alignment' column exists
        if 'alignment' not in self.df.columns:
            print("Error: 'alignment' column not found in data.")
            return

        # Filter for "Aligned"
        # Note: The alignment values are typically "Aligned", "Diverged", or "Neutral" / "N/A"
        df_aligned = self.df[self.df['alignment'] == 'Aligned'].copy()
        
        count = len(df_aligned)
        print(f"Found {count} aligned records.")
        
        if count > 0:
            # Construct output path
            output_file = os.path.join(self.output_dir, "aligned_records_only.csv")
            
            # Save to CSV
            df_aligned.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"✅ Extracted aligned records saved to: {output_file}")
            
            # Optional: Print a preview
            print("\nPreview of Aligned Records (Top 5):")
            print(df_aligned[['date', 'company_name', 'news_sentiment', 'market_car_3d']].head())
            
        return df_aligned

if __name__ == "__main__":
    # Path assumption: Running from streamlit_app/advanced_analysis or similar
    # Adjust path to find the CSV in the parent directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, "export_mns_demo_enriched.csv")
    
    # Analyze in the advanced_analysis folder
    output_dir = os.path.dirname(os.path.abspath(__file__))
    
    extractor = AlignmentExtractor(csv_path, output_dir)
    extractor.extract_aligned_records()
