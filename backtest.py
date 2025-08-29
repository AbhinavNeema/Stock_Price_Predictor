import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta as ta
from datetime import timedelta

from models import SECTOR_MODELS
from utils import _safe_float

def _build_window_features(window_df, config):
    df = window_df.copy()
    for stock in config['companies']:
        if stock in df.columns:
            try:
                df[f'{stock}_SMA_20'] = ta.sma(df[stock], length=20)
                df[f'{stock}_RSI_14'] = ta.rsi(df[stock], length=14)
            except Exception:
                df[f'{stock}_SMA_20'] = np.nan
                df[f'{stock}_RSI_14'] = np.nan
            df[f'{stock}_Sentiment'] = 0.0
            df[f'{stock}_Return'] = 0.0
        else:
            df[stock] = np.nan
            df[f'{stock}_SMA_20'] = np.nan
            df[f'{stock}_RSI_14'] = np.nan
            df[f'{stock}_Sentiment'] = 0.0
            df[f'{stock}_Return'] = 0.0
    if 'training_columns' in config:
        for col in config['training_columns']:
            if col not in df.columns:
                df[col] = 0.0
        df = df[config['training_columns']].ffill().bfill()
    return df

def run_backtest_for_sector(config, start_date, end_date, window=60, max_points=None):
    result = {'per_ticker': {}, 'summary': {}}
    fetch_start = pd.to_datetime(start_date) - pd.Timedelta(days=window * 3)
    fetch_end = pd.to_datetime(end_date) + pd.Timedelta(days=2)
    try:
        raw = yf.download(config['features_to_download'], start=fetch_start, end=fetch_end, progress=False, threads=False)
        if 'Close' in raw:
            price_df = raw['Close'].copy()
        else:
            price_df = raw.copy()
    except Exception as e:
        return {'error': f"Failed to download historical data: {e}"}
    if config.get('rename_map'):
        price_df.rename(columns=config['rename_map'], inplace=True)
    price_df.ffill(inplace=True)
    price_df.dropna(how='all', inplace=True)
    price_df = price_df.sort_index()
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    n_rows = len(price_df)
    if n_rows < window + 1:
        return {'error': f"Not enough historical rows ({n_rows}) for window={window}"}
    valid_indices = [i for i, d in enumerate(price_df.index) if (d >= start_date and d <= end_date)]
    if not valid_indices:
        return {'error': 'No data in requested date range'}

    for ticker in config['companies']:
        preds = []
        actuals = []
        dates = []
        try:
            col_present = ticker in price_df.columns
            if not col_present:
                result['per_ticker'][ticker] = {'error': 'Ticker missing from downloaded data'}
                continue
        except Exception:
            result['per_ticker'][ticker] = {'error': 'Ticker missing from downloaded data'}
            continue

        for j in range(window - 1, len(price_df) - 1):
            window_start_idx = j - (window - 1)
            window_end_idx = j
            last_day = price_df.index[window_end_idx]
            if not (last_day >= start_date and last_day <= end_date):
                continue
            if max_points and len(preds) >= max_points:
                break
            window_df = price_df.iloc[window_start_idx:window_end_idx + 1].copy()
            window_features = _build_window_features(window_df, config)
            try:
                ordered_data = window_features[config['training_columns']]
            except Exception as e:
                print(f"Backtest: missing training columns for {ticker}: {e}")
                continue
            try:
                scaled_data = config['scaler'].transform(ordered_data)
                df_scaled = pd.DataFrame(scaled_data, columns=ordered_data.columns)
                X_pred = np.array([df_scaled[config['feature_cols']].values])
                all_pred_scaled = config['model'].predict(X_pred)
                dummy_array = np.zeros((1, len(config['training_columns'])))
                target_indices = [config['training_columns'].index(t) for t in config['target_cols']]
                for idx_i, idx in enumerate(target_indices):
                    dummy_array[0, idx] = all_pred_scaled[0, idx_i]
                unscaled = config['scaler'].inverse_transform(dummy_array)
                unscaled_returns = unscaled[0, target_indices]
                target_name = f"{ticker}_Return"
                if target_name in config['training_columns']:
                    target_pos = config['target_cols'].index(target_name) if target_name in config['target_cols'] else None
                    if target_pos is not None:
                        predicted_return = float(unscaled_returns[target_pos])
                    else:
                        company_index = config['companies'].index(ticker)
                        predicted_return = float(unscaled_returns[company_index]) if company_index < len(unscaled_returns) else 0.0
                else:
                    company_index = config['companies'].index(ticker)
                    predicted_return = float(unscaled_returns[company_index]) if company_index < len(unscaled_returns) else 0.0
                last_close = float(window_df[ticker].iloc[-1])
                predicted_price = last_close * (1 + predicted_return)
                actual_next = float(price_df[ticker].iloc[window_end_idx + 1])
                preds.append(_safe_float(predicted_price))
                actuals.append(_safe_float(actual_next))
                dates.append(price_df.index[window_end_idx + 1].strftime('%Y-%m-%d'))
            except Exception as e:
                print(f"Backtest prediction error for {ticker} on {price_df.index[window_end_idx]}: {e}")
                continue
        if len(preds) == 0:
            result['per_ticker'][ticker] = {'error': 'No predictions (insufficient data)'}
            continue
        preds_arr = np.array(preds)
        actuals_arr = np.array(actuals)
        errors = preds_arr - actuals_arr
        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        with np.errstate(divide='ignore', invalid='ignore'):
            mape = float(np.mean(np.abs(errors / np.where(actuals_arr == 0, np.nan, actuals_arr))) * 100.0)
        if len(preds_arr) > 1:
            pred_deltas = np.sign(preds_arr[1:] - preds_arr[:-1])
            act_deltas = np.sign(actuals_arr[1:] - actuals_arr[:-1])
            direction_accuracy = float(np.mean(pred_deltas == act_deltas) * 100.0)
        else:
            direction_accuracy = 100.0 if (np.sign(preds_arr[0] - actuals_arr[0]) == np.sign(actuals_arr[0] - actuals_arr[0])) else 0.0
        result['per_ticker'][ticker] = {
            'count': int(len(preds)),
            'mae': round(mae, 4),
            'rmse': round(rmse, 4),
            'mape_percent': round(mape, 4) if not np.isnan(mape) else None,
            'direction_accuracy_percent': round(direction_accuracy, 2),
            'dates': dates,
            'predicted_prices': [round(float(p), 2) for p in preds],
            'actual_prices': [round(float(a), 2) for a in actuals]
        }
    all_mae = []
    all_rmse = []
    all_counts = 0
    for t, info in result['per_ticker'].items():
        if info.get('count'):
            all_counts += info['count']
            all_mae.append(info['mae'])
            all_rmse.append(info['rmse'])
    result['summary'] = {
        'tickers_tested': len(result['per_ticker']),
        'total_predictions': int(all_counts),
        'avg_mae': round(float(np.mean(all_mae)) , 4) if all_mae else None,
        'avg_rmse': round(float(np.mean(all_rmse)) , 4) if all_rmse else None
    }
    return result
