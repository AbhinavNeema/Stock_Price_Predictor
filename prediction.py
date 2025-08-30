import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
import traceback
from scipy.stats import norm

def _get_sector_predictions_base(config):
    """
    Fetches live data, calculates features, and returns predictions for a sector.
    This version includes robust error logging to diagnose silent failures.
    """
    companies = config.get('companies', [])
    model = config.get('model')
    scaler = config.get('scaler')
    features = config.get('features', [])

    if not all([companies, model, scaler, features]):
        print(f"  - ❌ ERROR: Configuration for sector is incomplete. Missing model, scaler, companies, or features.")
        return None, None

    # --- ADDED: This entire try...except block provides detailed debugging ---
    try:
        print(f"  - Attempting to download data for: {companies}")
        end_date = pd.Timestamp.now(tz='UTC')
        start_date = end_date - pd.Timedelta(days=365)
        
        df = yf.download(companies, start=start_date, end=end_date, progress=False)
        
        if df.empty:
            print(f"  - ❌ ERROR: yfinance returned an empty DataFrame for {companies}. Check if ticker symbols are valid.")
            return None, None
            
        live_df = df['Adj Close'].copy().dropna(how='all')
        
        if len(live_df) < 60:
             print(f"  - ❌ ERROR: Not enough historical data for {companies} to make a prediction. (Need at least 60 days, have {len(live_df)})")
             return None, None

        print("  - Calculating technical indicators...")
        # NOTE: Ensure the features in your config match the indicators calculated here.
        # This is a common source of errors.
        for ticker in companies:
            if ticker in live_df:
                live_df.ta.rsi(close=live_df[ticker], length=14, append=True)
                live_df.ta.ema(close=live_df[ticker], length=50, append=True)
                # Add any other indicators your model requires here

        live_df.dropna(inplace=True)

        if live_df.empty:
            print("  - ❌ ERROR: DataFrame became empty after calculating indicators and dropping NaNs. There might be issues with the source data.")
            return None, None

        print("  - Scaling features and making prediction...")
        latest_features_df = live_df[features]
        # Ensure columns are in the correct order, another common error source
        latest_features_df = latest_features_df[features] 
        
        latest_features = latest_features_df.iloc[-1:].values
        scaled_features = scaler.transform(latest_features)
        
        predicted_scaled_returns = model.predict(scaled_features)
        
        # This part is tricky and depends on how your scaler was trained.
        # This assumes the return column was the first one.
        pad_width = len(features) - predicted_scaled_returns.shape[1]
        padded_prediction = np.concatenate([predicted_scaled_returns, np.zeros((predicted_scaled_returns.shape[0], pad_width))], axis=1)
        unscaled_returns = scaler.inverse_transform(padded_prediction)[:, 0]

        return live_df, unscaled_returns

    except Exception as e:
        # This will catch any unexpected error and print the full traceback to your logs
        print(f"  - ❌ CRITICAL ERROR in _get_sector_predictions_base for sector '{config.get('name', 'Unknown')}': {e}")
        traceback.print_exc() 
        return None, None

def estimate_probability_of_reaching(predicted_return, returns_series):
    """
    Estimates the probability of a stock reaching its predicted return
    based on its recent historical volatility.
    """
    if returns_series.empty or returns_series.std() == 0:
        return 0.5 # Default to 50% chance if no volatility data is available
    
    mean_return = returns_series.mean()
    std_dev = returns_series.std()
    
    # Calculate the Z-score, which measures how many standard deviations the predicted return is from the mean.
    z_score = (predicted_return - mean_return) / std_dev
    
    # Use the cumulative distribution function (CDF) of a normal distribution
    # to find the probability of a value being less than or equal to the Z-score.
    if predicted_return >= mean_return:
        # Probability of achieving a return greater than or equal to the prediction
        return 1 - norm.cdf(z_score)
    else:
        # Probability of achieving a return less than or equal to the prediction
        return norm.cdf(z_score)
