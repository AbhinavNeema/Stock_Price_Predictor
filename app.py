import threading
import os
import time
from flask import Flask, request, jsonify
from flask_cors import CORS

# Import functions and the shared state object from your modules
# (we import setup_scheduler but will call it after models are loaded)
from models import load_all_best_models, SECTOR_MODELS
from scheduler import setup_scheduler, run_daily_predictions, state
from backtest import run_backtest_for_sector
from prediction import _get_sector_predictions_base, estimate_probability_of_reaching
from utils import _safe_float

app = Flask(__name__)
CORS(app)

# --- readiness / background model loading ---
_models_ready = threading.Event()

def _load_models_and_start_scheduler():
    """Load models, then setup scheduler once models are available."""
    try:
        print("🔁 Starting background model load...")
        load_all_best_models()
        print("✅ Models loaded.")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
    finally:
        # Mark models as ready (even if loading failed; you can inspect logs)
        _models_ready.set()
        # Start scheduler only after models attempt to load (prevents jobs failing)
        try:
            print("🔁 Starting scheduler...")
            setup_scheduler()
            print("✅ Scheduler started.")
        except Exception as e:
            print(f"❌ Scheduler failed to start: {e}")

# Start loading models in background thread
threading.Thread(target=_load_models_and_start_scheduler, daemon=True).start()

# -----------------------
# Routes (kept same logic as you provided)
# -----------------------

@app.route('/config', methods=['GET'])
def get_config():
    frontend_config = {sector: {t: t.split('.')[0] for t in d['companies']} for sector, d in SECTOR_MODELS.items()}
    return jsonify(frontend_config)


@app.route('/daily-predictions', methods=['GET'])
def get_daily_predictions_api():
    return jsonify({"status": state.prediction_status, "predictions": state.daily_predictions})


@app.route('/top-stocks', methods=['GET'])
def get_top_stocks():
    print("\n--- Serving Top 5 Stock Analysis ---")

    if not state.daily_predictions and not state.prediction_lock.locked():
        threading.Thread(target=run_daily_predictions, daemon=True).start()

    if not state.daily_predictions:
        print("Predictions not ready. Waiting for job to complete...")
        state.prediction_ready_event.wait(timeout=60)

    if not state.daily_predictions:
        return jsonify({'error': 'Predictions are not available. Please try again later.'}), 503

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
    data = request.get_json() or {}
    sector = data.get('sector')
    ticker = data.get('ticker')
    if not sector or not ticker or sector not in SECTOR_MODELS:
        return jsonify({'error': 'Invalid sector or company specified.'}), 400

    if not state.daily_predictions:
        state.prediction_ready_event.wait(timeout=10)

    cached = state.daily_predictions.get(ticker)
    if cached and cached.get('sector') == sector:
        return jsonify({
            'predicted_price': cached.get('predictedPrice'),
            'current_price': cached.get('currentPrice'),
            'predictedChangePercent': cached.get('predictedChangePercent'),
            'predictedChancePercent': cached.get('predictedChancePercent'),
            'history': cached.get('history')
        })

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
    if state.prediction_lock.locked():
        return jsonify({"message": "A prediction job is already running."}), 202
    threading.Thread(target=run_daily_predictions, daemon=True).start()
    return jsonify({"message": "Prediction job started in background."}), 202


@app.route('/health', methods=['GET'])
def health():
    """Return 200 only after models are loaded (helps readiness checks)."""
    if not _models_ready.is_set():
        return jsonify({'status': 'loading models'}), 503
    return jsonify({'status': 'ok'}), 200


# Local run / fallback (keeps app object for gunicorn: app)
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    # If a PORT env is present (production), prefer to run gunicorn if available, else fallback to Flask dev server
    if os.environ.get('PORT'):
        try:
            from gunicorn.app.wsgiapp import WSGIApplication
            class GunicornApp(WSGIApplication):
                def init(self, parser, opts, args):
                    return {'bind': f"0.0.0.0:{port}", 'workers': 1}
                def load(self):
                    return app
            print(f"Starting Gunicorn on 0.0.0.0:{port}")
            GunicornApp().run()
        except Exception as e:
            print(f"Gunicorn not available or failed to start ({e}). Falling back to Flask dev server.")
            app.run(host='0.0.0.0', port=port, debug=False)
    else:
        app.run(host='0.0.0.0', port=port, debug=True)
