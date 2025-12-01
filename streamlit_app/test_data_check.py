import yfinance as yf
import pandas as pd

tickers = ["6752.T", "^TOPX"]
start_date = "2025-10-30"
end_date = "2025-11-02"

print(f"Querying {tickers} from {start_date} to {end_date}...")

for t in tickers:
    try:
        stock = yf.Ticker(t)
        hist = stock.history(start=start_date, end=end_date)
        print(f"\n--- {t} ---")
        if hist.empty:
            print("No data.")
        else:
            print(hist[['Open', 'Close', 'Volume']])
    except Exception as e:
        print(f"Error {t}: {e}")
