# Stock Price Predictor

An end-to-end system that predicts stock price movements by combining technical analysis with live sentiment data from news. This deep-learning-based tool forecasts trends for major Indian stocks, delivering actionable insights for investors and analysts.

---

## 🚀 Overview

This project features:

- **Automated Daily Predictions**: A background scheduler (apscheduler) fetches data and generates next-day predictions for tracked stocks daily at 4:00 PM IST.
- **Performance-Optimized Caching**: Predictions are cached in memory for fast, low-latency API responses.
- **Live Sentiment Analysis**: Integrates real-time sentiment scores derived from recent news using the VADER sentiment tool.
- **Top Stock Ranking**: Ranks stocks using a custom metric:  
  **Score = Predicted Return / Volatility**
- **On-Demand Predictions**: Request real-time, single-stock predictions with historical data.
- **Modular Architecture**: Organized structure with dedicated directories for models, scalers, and scripts for scalability and maintainability.

---

## 📊 Methodology & Implementation

### Data Collection & Preprocessing
- Historical data from **yfinance**
- Technical indicators (e.g., SMA, RSI) via **pandas_ta**
- Sentiment data processed in `sentimentFinal.ipynb`, using news APIs and VADER

### Model Architecture
- Sector-specific Keras models saved in `models/`; scalers in `scalers/`
- Key modules:
  - `models.py` & `prediction.py`: model loading and prediction logic
  - `backtest.py`: historical model performance evaluation
  - `scheduler.py`: manages automated daily predictions
  - `utils.py`: helper utilities (data fetch, feature engineering, sentiment helpers)

---

## 📂 Project Structure

    ├── models/               # Trained Keras model files
    ├── scalers/              # Scaler files
    ├── .gitignore
    ├── app.py                # Flask API endpoints & server
    ├── backtest.py           # Historical performance testing
    ├── index.html            # Optional web interface
    ├── models.py             # Utility functions for models
    ├── prediction.py         # Prediction operations
    ├── README.md             # Project documentation
    ├── requirements.txt      # Python dependencies
    ├── scheduler.py          # Scheduler for daily predictions
    ├── sentimentFinal.ipynb  # Sentiment analysis notebook
    └── utils.py              # Helper utilities

---

## ⚙️ Installation & Usage

### 1. Clone the repository

    git clone https://github.com/AbhinavNeema/Stock_Price_Predictor.git
    cd Stock_Price_Predictor

### 2. Set up virtual environment and install dependencies

    python -m venv venv
    # On Linux/macOS:
    source venv/bin/activate
    # On Windows:
    venv\Scripts\activate
    pip install -r requirements.txt

### 3. Configure environment variables
Create a `.env` file at the project root and add required keys (example):

    NEWS_API_KEY=your_news_api_key_here
    OTHER_SECRET=your_other_secret_here

> **Security note:** Do not commit `.env` or any secrets to the repo. Add `.env` to `.gitignore`.

### 4. Run the application

    # ensure virtualenv is activated
    python app.py

The Flask server will start (default: `http://127.0.0.1:5001` or as configured) and expose the API endpoints listed below.

---

## 🔗 API Endpoints

| Endpoint              | Method | Description                                                  |
|-----------------------|--------|--------------------------------------------------------------|
| `/daily-predictions`  | GET    | Returns cached daily predictions for all tracked stocks      |
| `/top-stocks`         | GET    | Returns top 5 stocks based on risk-adjusted ranking         |
| `/predict`            | POST   | Generates real-time prediction for a single stock           |
| `/backtest`           | POST   | Runs backtest for a chosen sector over a selected timeframe |
| `/config`             | GET    | Retrieves current config and available sectors              |

### Example: `/predict` request (HTTP)

    POST /predict
    Content-Type: application/json

    {
      "ticker": "TCS.NS",
      "start_date": "2024-01-01",
      "end_date": "2024-12-31",
      "window": 60
    }

### Example: `/predict` response (JSON)

    {
      "ticker": "TCS.NS",
      "predicted_return": 1.5,
      "predicted_price": 3420.25,
      "volatility": 12.5,
      "score": 0.12,
      "timestamp": "2025-08-29T16:00:00+05:30"
    }

### Example: `/top-stocks` response (JSON)

    [
      {"ticker": "TCS.NS", "score": 0.12, "predicted_return": 1.5, "volatility": 12.5},
      {"ticker": "HDFCBANK.NS", "score": 0.09, "predicted_return": 1.2, "volatility": 13.3},
      {"ticker": "INFY.NS", "score": 0.08, "predicted_return": 1.1, "volatility": 13.8},
      {"ticker": "HINDUNILVR.NS", "score": 0.06, "predicted_return": 0.9, "volatility": 15.0},
      {"ticker": "SUNPHARMA.NS", "score": 0.05, "predicted_return": 0.8, "volatility": 16.0}
    ]

---

## 📈 Backtesting

Run historical evaluation using `backtest.py`:

    python backtest.py --sector IT --start 2020-01-01 --end 2024-01-01 --window 60

Adjust `--window` (walk-forward window), `--max_points`, and other flags as implemented in the script.

**Notes on backtesting:**
- Uses walk-forward evaluation (`run_backtest_for_sector(config, start_date, end_date, window=60, max_points=None)`).
- Historical sentiment is set to 0 by default unless you pass precomputed sentiment series.
- Use real historical sentiment for more realistic results if available.

---

## 🧪 Tests & Troubleshooting

- If models fail to load, confirm correct model file names in `models/` and corresponding scaler files in `scalers/`.
- If you see scaler version mismatch warnings, recreate scalers with your runtime scikit-learn version or match the version used when saving.
- For TensorFlow retracing warnings: ensure model-building code uses consistent input shapes and avoid rebuilding models in loops.
- Check logs printed by `app.py` and `scheduler.py` for detailed error traces.

---

## ⏭ Future Enhancements

- **Real-time Data Integration**: Add websocket or market-feed ingestion for live tick-level updates.
- **Web UI**: Build an interactive dashboard with charts, filters and model explainability.
- **Model Improvements**: Experiment with Transformers, attention-based models or ensemble methods.
- **Broader Coverage**: Expand to more sectors, more tickers and multi-exchange support.

---

## 📬 Contact

Questions or suggestions? Reach out:

**Abhinav Neema** — abhinavneema22@gmail.com

---