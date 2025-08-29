import threading
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from gunicorn.app.wsgiapp import WSGIApplication

# Import functions and the shared state object from the other modules
# Assuming these files are in the same directory for deployment on Render.
from models import load_all_best_models, SECTOR_MODELS
from scheduler import setup_scheduler, run_daily_predictions, state
from backtest import run_backtest_for_sector
from prediction import _get_sector_predictions_base, estimate_probability_of_reaching
from utils import _safe_float

app = Flask(__name__)
CORS(app)

# --- 1. Setup and Model Loading ---
# This function loads the pre-trained models from the 'models.py' module.
load_all_best_models()

# --- 2. Daily Prediction Cache & Scheduler Setup ---
# This function sets up the background scheduler from the 'scheduler.py' module
# that will run the daily prediction job.
setup_scheduler()

# --- 3. API Endpoints ---

@app.route('/config', methods=['GET'])
def get_config():
    """Endpoint to return the configuration of sectors and companies."""
    frontend_config = {sector: {t: t.split('.')[0] for t in d['companies']} for sector, d in SECTOR_MODELS.items()}
    return jsonify(frontend_config)


@app.route('/daily-predictions', methods=['GET'])
def get_daily_predictions_api():
    """Returns the pre-calculated stock price predictions for the day."""
    return jsonify({"status": state.prediction_status, "predictions": state.daily_predictions})


@app.route('/top-stocks', methods=['GET'])
def get_top_stocks():
    """
    Returns the top 5 stocks based on a risk-adjusted score.
    It waits for the prediction job to finish if it's currently running.
    """
    print("\n--- Serving Top 5 Stock Analysis ---")

    # If the cache is empty, start a non-blocking background thread to run the predictions.
    # The `prediction_lock` ensures we only start one.
    if not state.daily_predictions and not state.prediction_lock.locked():
        threading.Thread(target=run_daily_predictions, daemon=True).start()

    # Wait for the prediction job to complete with a timeout.
    # This ensures the API remains responsive.
    if not state.daily_predictions:
        print("Predictions not ready. Waiting for job to complete...")
        state.prediction_ready_event.wait(timeout=60) # Wait up to 60 seconds

    # If the job failed or returned no data, handle gracefully.
    if not state.daily_predictions:
        return jsonify({'error': 'Predictions are not available. Please try again later.'}), 503

    # Build metrics from the now-populated cache.
    all_stocks_metrics = []
    for ticker, info in state.daily_predictions.items():
        predicted_return = _safe_float(info.get('predictedReturn', 0.0))
        volatility = _safe_float(info.get('volatility', 0.0))
        score = float(predicted_return / volatility) if volatility and volatility > 0 else 0.0
        all_stocks_metrics.append({
            'ticker': ticker,
            'sector': info.get('sector'),
            'predictedReturnPercent': round(predicted_return * 100, 2),
            'riskVolatilityPercent': round(volatility * 100, 2),
            'score': round(score, 2),
            'currentPrice': _safe_float(info.get('currentPrice')),
            'predictedPrice': _safe_float(info.get('predictedPrice')),
            'predictedChangePercent': _safe_float(info.get('predictedChangePercent')),
            'predictedChancePercent': _safe_float(info.get('predictedChancePercent'))
        })
    sorted_stocks = sorted(all_stocks_metrics, key=lambda x: x['score'], reverse=True)
    top5 = sorted_stocks[:5]
    print("--- Top 5 Stock Analysis Complete ---")
    return jsonify({'status': state.prediction_status, 'top5': top5})


@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint for single, on-demand predictions.
    Uses cached data when available for speed, otherwise performs a live calculation.
    """
    data = request.get_json() or {}
    sector = data.get('sector')
    ticker = data.get('ticker')
    if not sector or not ticker or sector not in SECTOR_MODELS:
        return jsonify({'error': 'Invalid sector or company specified.'}), 400

    # First, try to get the data from the pre-calculated cache.
    # We wait a short period to see if the cache is being populated.
    if not state.daily_predictions:
        state.prediction_ready_event.wait(timeout=10) # Wait up to 10 seconds for the cache to be ready

    cached = state.daily_predictions.get(ticker)
    if cached and cached.get('sector') == sector:
        return jsonify({
            'predicted_price': cached.get('predictedPrice'),
            'current_price': cached.get('currentPrice'),
            'predictedChangePercent': cached.get('predictedChangePercent'),
            'predictedChancePercent': cached.get('predictedChancePercent'),
            'history': cached.get('history')
        })

    # If not in the cache, or the cache is old/stale, do a live calculation.
    config = SECTOR_MODELS[sector]
    try:
        live_df, unscaled_returns = _get_sector_predictions_base(config)
        if live_df is None or unscaled_returns is None:
            return jsonify({'error': 'Not enough recent data to calculate features and predict.'}), 400
        company_index = config['companies'].index(ticker)
        predicted_return = _safe_float(unscaled_returns[company_index])
        last_close_price = _safe_float(live_df[ticker].iloc[-1])
        predicted_price = _safe_float(last_close_price * (1 + predicted_return))
        returns_series = live_df[ticker].pct_change().tail(60).dropna()
        chance = estimate_probability_of_reaching(predicted_return, returns_series)
        chance_pct = _safe_float(chance * 100.0)
        hist_series = live_df[ticker].tail(90).dropna()
        history_formatted = {
            'labels': [d.strftime('%Y-%m-%d') for d in hist_series.index],
            'prices': [round(_safe_float(p), 2) for p in hist_series.values]
        }
        return jsonify({
            'predicted_price': round(predicted_price, 2),
            'current_price': round(last_close_price, 2),
            'predictedChangePercent': round(predicted_return * 100, 2),
            'predictedChancePercent': round(chance_pct, 2),
            'history': history_formatted
        })
    except Exception as e:
        print(f"Error during prediction for {ticker} in {sector}: {e}")
        return jsonify({'error': 'An error occurred during prediction.'}), 500

@app.route('/backtest', methods=['POST'])
def backtest_api():
    """Endpoint to run a historical backtest."""
    data = request.get_json() or {}
    sector = data.get('sector')
    ticker = data.get('ticker')
    start_date = data.get('start_date')
    end_date = data.get('end_date')
    window = int(data.get('window', 60))
    max_points = data.get('max_points', None)
    if not sector or sector not in SECTOR_MODELS:
        return jsonify({'error': 'Invalid or missing sector.'}), 400
    if not start_date or not end_date:
        return jsonify({'error': 'Please provide start_date and end_date.'}), 400
    config = SECTOR_MODELS[sector]
    if ticker:
        if ticker not in config['companies']:
            return jsonify({'error': 'Ticker not part of the specified sector.'}), 400
        config_for_run = dict(config)
        config_for_run['companies'] = [ticker]
    else:
        config_for_run = config
    try:
        res = run_backtest_for_sector(config_for_run, start_date, end_date, window=window, max_points=max_points)
        res['_note'] = ("Historical backtest sets per-company sentiment to 0 (no historical news). "
                        "If you have saved historical sentiment, replace that column in historical prices before running backtest.")
        return jsonify(res)
    except Exception as e:
        return jsonify({'error': f'Backtest failed: {e}'}), 500


@app.route("/run-now", methods=["POST"])
def run_now():
    """Endpoint to manually trigger the daily prediction job."""
    if state.prediction_lock.locked():
        return jsonify({"message": "A prediction job is already running."}), 202
    threading.Thread(target=run_daily_predictions, daemon=True).start()
    return jsonify({"message": "Prediction job started in background."}), 202

# The following lines are for deployment on Render, they are not needed when running locally
# and are a recommended alternative to `if __name__ == '__main__':`.
# They instruct Gunicorn to use the Flask app object.
class GunicornApp(WSGIApplication):
    def init(self, parser, opts, args):
        return {
            'bind': f"0.0.0.0:{os.environ.get('PORT', '5001')}",
            'workers': 1
        }
    def load(self):
        return app

if __name__ == '__main__':
    GunicornApp().run()
