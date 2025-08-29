import math
import time
import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta as ta

from models import SECTOR_MODELS
from utils import _safe_float, get_live_sentiment

def _get_sector_predictions_base(config):
    """
    Base function to download data, process features, and run the model.
    Returns the live data dataframe and the unscaled predicted returns.
    """
    try:
        end_date = pd.to_datetime('today')
        start_date = end_date - pd.Timedelta(days=180)
        live_data_raw = yf.download(config['features_to_download'], start=start_date, end=end_date, progress=False, threads=False)
        if 'Close' in live_data_raw:
            live_df = live_data_raw['Close'].copy()
        else:
            live_df = live_data_raw.copy()
        if config.get('rename_map'):
            live_df.rename(columns=config['rename_map'], inplace=True)
        live_df.ffill(inplace=True)

        for stock in config['companies']:
            if stock not in live_df.columns:
                print(f"  - {stock} not present in downloaded data for {config['sector_name']}")
                continue
            try:
                live_df[f'{stock}_SMA_20'] = ta.sma(live_df[stock], length=20)
                live_df[f'{stock}_RSI_14'] = ta.rsi(live_df[stock], length=14)
            except Exception:
                pass
            print(f"  Fetching sentiment for {stock}...")
            live_df[f'{stock}_Sentiment'] = get_live_sentiment(stock)
            live_df[f'{stock}_Return'] = 0.0
            time.sleep(1)

        live_df.dropna(inplace=True)
        if len(live_df) < 60:
            return None, None

        last_60_days = live_df.tail(60)
        ordered_data = last_60_days[config['training_columns']]
        scaled_data = config['scaler'].transform(ordered_data)
        df_scaled = pd.DataFrame(scaled_data, columns=ordered_data.columns)
        X_pred = np.array([df_scaled[config['feature_cols']].values])
        all_predicted_returns_scaled = config['model'].predict(X_pred)
        dummy_array = np.zeros((1, len(config['training_columns'])))
        target_indices = [config['training_columns'].index(t) for t in config['target_cols']]
        for i, idx in enumerate(target_indices):
            dummy_array[0, idx] = all_predicted_returns_scaled[0, i]

        unscaled_predictions = config['scaler'].inverse_transform(dummy_array)
        unscaled_returns = unscaled_predictions[0, target_indices]
        return live_df, unscaled_returns
    except Exception as e:
        print(f"  - Error in _get_sector_predictions_base for {config.get('sector_name','unknown')}: {e}")
        return None, None

def estimate_probability_of_reaching(predicted_return, returns_series):
    """Estimate probability that next-day return >= predicted_return."""
    try:
        rs = returns_series.dropna()
        if rs.empty:
            return 0.0
        mu = float(rs.mean())
        sigma = float(rs.std())
        if sigma <= 0:
            return 1.0 if mu >= predicted_return else 0.0
        z = (predicted_return - mu) / sigma
        cdf = 0.5 * (1 + math.erf(z / math.sqrt(2)))
        prob = 1.0 - cdf
        return max(0.0, min(1.0, prob))
    except Exception:
        return 0.0
