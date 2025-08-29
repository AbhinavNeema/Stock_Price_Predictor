# A Multi-Sector Stock Price Predictive Model

![Python](https://img.shields.io/badge/python-3.12-blue?logo=python)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Build](https://img.shields.io/badge/build-passing-brightgreen)

---

## Overview
This project is a **powerful predictive model** designed to analyze and forecast key financial metrics across major economic sectors. By leveraging advanced **machine learning models** trained on historical data, this system provides actionable insights for each designated sector.  

The core innovation of this project lies in its **automated daily prediction engine** and the integration of **live market sentiment**, providing a comprehensive, data-driven tool for strategic market analysis and decision-making.

---

## Key Features

- **Automated Daily Predictions**  
  Uses a background scheduler (`apscheduler`) to automatically fetch data and generate **next-day stock price predictions** for all tracked stocks every day at **4:00 PM IST**.

- **Performance-Optimized Caching**  
  Daily predictions are cached in memory to ensure **fast, low-latency API responses**.

- **Live Sentiment Analysis**  
  Incorporates a live sentiment score for each stock, derived from recent news articles using the **VADER sentiment analysis tool**.

- **Top Stock Ranking**  
  Provides a **top 5 stock ranking** based on a custom metric:  
Score = Predicted Return / Volatility

- **On-Demand Predictions**  
Users can request **real-time, single-stock predictions** with historical price data for immediate analysis.

- **Modular Architecture**  
Clear separation of concerns with **dedicated directories** for models, scalers, and scripts, making it scalable and easy to maintain.

---

## Methodology and Implementation

### Data and Preprocessing
- Historical stock data is fetched from **yfinance**.  
- Technical indicators such as **SMA** and **RSI** are generated using **pandas_ta**.  
- Sentiment data is sourced from a **news API**, processed in `sentimentFinal.ipynb`, and used to enhance model predictions.

### Model Architecture
- **Multi-model architecture**: Each economic sector has a specialized **Keras model**.  
- Models are stored in the `models/` directory; corresponding scalers are stored in `scalers/`.  
- Core logic for model loading and prediction execution resides in `models.py` and `prediction.py`.  
- **Backtesting** is implemented in `backtest.py` to evaluate historical performance.  
- **Scheduler** (`scheduler.py`) manages automated daily predictions.

---

## Project Structure
.
├── .vscode/ # VS Code configuration files
├── models/ # Directory for trained Keras model files
├── scalers/ # Directory for saved data scaler files
├── venv/ # Python virtual environment
├── .env # Environment variables (API keys, etc.)
├── .gitignore # Git ignore file
├── app.py # Main application with API endpoints
├── backtest.py # Script for backtesting models
├── index.html # Web interface (optional)
├── models.py # Model-related utility functions
├── prediction.py # Core prediction logic
├── readme.md # Project documentation
├── requirements.txt # Python dependencies
├── scheduler.py # Daily prediction scheduler script
├── sentimentFinal.ipynb# Jupyter notebook for sentiment analysis exploration
└── utils.py # Utility functions

---

## Installation & Usage

### 1. Clone the Repository
```bash
git clone [repository_url]
cd [project_folder_name]
2. Install Dependencies
It is recommended to use a virtual environment:
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows

pip install -r requirements.txt
3. Run the Application
python app.py
The application will start a Flask server with API endpoints for predictions and top-stock ranking.
Future Enhancements
Real-Time Data Integration: Incorporate APIs to provide live, up-to-the-minute predictions.
Web Interface: User-friendly front-end to visualize predictions, top stocks, and historical performance.
Model Optimization: Explore more advanced architectures (e.g., Transformers, hybrid LSTM-CNN) for improved performance.
Extended Sector Coverage: Add additional economic sectors and stocks for broader market insights.
API Endpoints
Endpoint	Method	Description
/daily-predictions	GET	Returns cached predictions for all stocks.
/top-stocks	GET	Returns the top 5 stocks by risk-adjusted score.
/predict	POST	Returns a single-stock prediction given historical data.
/backtest	POST	Performs a backtest for a specified sector and date range.
/config	GET	Returns current system configuration and available sectors.
Contact
For questions or suggestions, please contact [Your Name] at [Your Email Address].