# MNS News Analysis Dashboard

A Streamlit-based dashboard for analyzing and validating financial news sentiment, significance, and market impact. This tool allows analysts to review model outputs, validate predictions against market data, and provide feedback for model improvement.

## Key Features

### 🔍 Discovery & Filtering
*   **Global Search:** Instantly search across headlines, reasoning, key factors, and tickers.
*   **Advanced Filters:** Filter by Ticker, Date Range, Sentiment, Significance, and Event Correlation tags.
*   **Favorites:** "Star" interesting news items to save them to a personal watchlist.

### 📊 Market Impact Analysis
*   **Real-Time Data:** Fetches market data using `yfinance` to show price movements on the day of the event.
*   **Key Metrics:**
    *   **Stock Chg:** % Change on the day.
    *   **Index Chg:** % Change of the benchmark index (TOPIX/ETF).
    *   **Rel Change:** Stock performance relative to the index.
    *   **Vol Ratio:** Volume relative to the 30-day average.

### 📝 Feedback Loop & Validation
*   **Model Validation:** Interactive form to validate:
    *   **Sentiment:** Correct/Incorrect (with correction dropdown).
    *   **Significance:** High/Low.
    *   **Source Quality:** Is the reporter useful?
    *   **Event Correlation:** Strength of the event link.
*   **Data Persistence:** Feedback is saved locally to `feedback_log.csv` and Favorites to `favorites.json`.

## Project Structure

```
vgeo-mns/
├── mns_demo_output.csv       # Source data (News events)
├── streamlit_app/
│   ├── app.py                # Main application logic
│   ├── data_manager.py       # Data loading, market data fetching, persistence
│   ├── requirements.txt      # Python dependencies
│   ├── run_app.sh            # Helper script to launch the app
│   ├── favorites.json        # Stores user favorites (auto-generated)
│   ├── feedback_log.csv      # Stores user feedback (auto-generated)
│   └── assets/
│       └── style.css         # Custom dark theme styling
└── README.md                 # This file
```

## Setup & Running Locally

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/verstandai/vgeo-mns.git
    cd vgeo-mns
    ```

2.  **Run the App:**
    You can use the helper script:
    ```bash
    cd streamlit_app
    ./run_app.sh
    ```
    Or run manually:
    ```bash
    pip install -r streamlit_app/requirements.txt
    streamlit run streamlit_app/app.py
    ```

## Deployment

This app is ready for deployment on **Streamlit Community Cloud**.

1.  Push your code to GitHub.
2.  Go to [share.streamlit.io](https://share.streamlit.io).
3.  Deploy a new app pointing to your repository.
4.  **Main file path:** `streamlit_app/app.py`

**Note on Persistence:**
In the "Quick & Dirty" deployment mode, `favorites.json` and `feedback_log.csv` are ephemeral and will reset when the app restarts. For production use, integrate with Google Sheets or a database.
