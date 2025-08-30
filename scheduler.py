import pandas as pd
import atexit
import threading
from apscheduler.schedulers.background import BackgroundScheduler

from models import SECTOR_MODELS
from prediction import _get_sector_predictions_base, estimate_probability_of_reaching
from utils import _safe_float

# --- Global State Object and Sync Primitives ---
# All global state is now contained in a single object to ensure it's shared across modules.
class State:
    def __init__(self):
        self.daily_predictions = {}
        self.prediction_status = {"status": "Not yet run", "last_updated": None}
        self.prediction_lock = threading.Lock()
        self.prediction_ready_event = threading.Event()

state = State()

def run_daily_predictions():
    """
    Scheduled job to calculate next-day prices for all stocks and cache them.
    This function is now responsible for setting up the lock and signaling completion.
    """
    # Acquire lock to prevent multiple jobs running concurrently
    if not state.prediction_lock.acquire(blocking=False):
        print("Prediction job is already running. Skipping this trigger.")
        return

    # Clear the event at the start to indicate a job is running
    state.prediction_ready_event.clear()

    try:
        print("\n--- 🚀 Starting Daily Stock Price Calculation ---")
        state.prediction_status = {"status": "running", "last_updated": pd.to_datetime('now', utc=True).isoformat()}
        all_predictions = {}
        for sector_name, config in SECTOR_MODELS.items():
            print(f"Processing sector for daily predictions: {sector_name}...")
            live_df, unscaled_returns = _get_sector_predictions_base(config)
            if live_df is None or unscaled_returns is None:
                print(f"  - Could not get data for sector {sector_name}. Skipping.")
                continue

            for i, stock_ticker in enumerate(config['companies']):
                try:
                    if stock_ticker not in live_df.columns:
                        print(f"  - {stock_ticker} not in live dataframe for sector {sector_name}, skipping.")
                        continue
                    predicted_return = _safe_float(unscaled_returns[i])
                    last_close_price = _safe_float(live_df[stock_ticker].iloc[-1])
                    predicted_price = _safe_float(last_close_price * (1 + predicted_return))
                    returns_series = live_df[stock_ticker].pct_change().tail(60).dropna()
                    volatility = _safe_float(returns_series.std(), default=0.0)
                    chance = estimate_probability_of_reaching(predicted_return, returns_series)
                    chance_pct = _safe_float(chance * 100.0)
                    hist_series = live_df[stock_ticker].tail(90).dropna()
                    history_formatted = {
                        'labels': [d.strftime('%Y-%m-%d') for d in hist_series.index],
                        'prices': [round(_safe_float(p), 2) for p in hist_series.values]
                    }
                    all_predictions[stock_ticker] = {
                        'sector': sector_name,
                        'predictedPrice': round(predicted_price, 2),
                        'currentPrice': round(last_close_price, 2),
                        'predictedChangePercent': round(predicted_return * 100, 2),
                        'predictedReturn': predicted_return,
                        'volatility': volatility,
                        'predictedChancePercent': round(chance_pct, 2),
                        'history': history_formatted
                    }
                except Exception as e:
                    print(f"  - Error preparing prediction entry for {stock_ticker}: {e}")
                    continue

        state.daily_predictions = all_predictions
        state.prediction_status = {"status": "completed", "last_updated": pd.to_datetime('now', utc=True).isoformat()}
        print(f"--- ✅ Daily Stock Price Calculation Complete. Cached {len(state.daily_predictions)} predictions. ---")
    finally:
        # Release the lock and set the event to signal completion
        state.prediction_lock.release()
        state.prediction_ready_event.set()

def setup_scheduler():
    scheduler = BackgroundScheduler(daemon=True, timezone='Asia/Kolkata')
    # Run once immediately on startup
    scheduler.add_job(run_daily_predictions, 'date')
    # Schedule to run every day at 4:00 PM (16:00) IST
    scheduler.add_job(run_daily_predictions, 'cron', hour=16, minute=0)
    scheduler.start()
    atexit.register(lambda: scheduler.shutdown())
