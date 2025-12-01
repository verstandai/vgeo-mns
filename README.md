# AlphaStream - Sentiment Analysis & Validation Tool

This project is a Streamlit-based dashboard designed to validate financial news sentiment analysis models. It allows analysts to review news events, visualize market impact, and provide structured feedback to retrain the underlying models.

## Project Structure

```text
vgeo-mns/
├── mns_demo_output.csv       # Source Data: The demo dataset containing news and sentiment scores.
├── README.md                 # Project documentation.
└── streamlit_app/            # The main application directory.
    ├── app.py                # Entry Point: The main Streamlit application script.
    ├── data_manager.py       # Data Layer: Handles loading CSVs, fetching market data (yfinance), and saving feedback.
    ├── requirements.txt      # Dependencies: Python packages required to run the app.
    ├── run_app.sh            # Helper Script: A shell script to install dependencies and run the app.
    ├── assets/               # Static Assets
    │   └── style.css         # Styling: Custom CSS for the "Financial Terminal" dark mode look.
    ├── .streamlit/           # Configuration: Streamlit-specific config folder (e.g., for themes).
    └── feedback_log.csv      # Output: Generated automatically to store user feedback.
```

## Key Components

*   **`app.py`**: The UI layer. It handles the layout, filters (Sidebar), and the rendering of news cards. It uses `data_manager.py` to fetch data.
*   **`data_manager.py`**: The logic layer.
    *   **`load_data()`**: Reads and cleans `mns_demo_output.csv`.
    *   **`get_market_data()`**: Fetches real-time or mock market data (Price, Volume, Index comparison) using `yfinance`.
    *   **`save_feedback()`**: Appends analyst inputs to `feedback_log.csv`.
*   **`mns_demo_output.csv`**: The input file. Contains columns like `headline`, `news_sentiment`, `classification`, `reasoning`, etc.

## How to Run

1.  Navigate to the app directory:
    ```bash
    cd streamlit_app
    ```

2.  Run the helper script (installs requirements and launches app):
    ```bash
    ./run_app.sh
    ```

    *Alternatively, run manually:*
    ```bash
    pip install -r requirements.txt
    streamlit run app.py
    ```

## Features

*   **Sentiment & Significance Badges**: Quick visual indicators for model outputs.
*   **Market Impact Analysis**: Visualizes stock price change relative to the index and volume anomalies.
*   **Analyst Feedback Loop**: A form to correct sentiment, flag duplicates, and provide strategy notes for model retraining.
