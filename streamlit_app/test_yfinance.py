import yfinance as yf
import pandas as pd

ticker = "PCRHY"
start_date = "2025-10-01"
end_date = "2025-10-31"

print(f"Querying {ticker} from {start_date} to {end_date}...")

try:
    stock = yf.Ticker(ticker)
    hist = stock.history(start=start_date, end=end_date)
    
    if hist.empty:
        print("RESULT: No data returned (DataFrame is empty).")
    else:
        print("RESULT: Data found!")
        print(hist.head())
except Exception as e:
    print(f"ERROR: {e}")
