import yfinance as yf
tickers = ["1306.T"]
start_date = "2025-10-31"
end_date = "2025-11-01"
stock = yf.Ticker(tickers[0])
hist = stock.history(start=start_date, end=end_date)
print(hist)
