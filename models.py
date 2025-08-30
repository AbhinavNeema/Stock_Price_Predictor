import os
import joblib
import traceback
from tensorflow.keras.models import load_model

SECTOR_MODELS = {}

def load_all_best_models():
    """Loads all the BEST trained sector models and their configs."""
    global SECTOR_MODELS
    configs = [
        {
            'sector_name': 'IT', 'companies': ['TCS.NS', 'INFY.NS', 'WIPRO.NS', 'HCLTECH.NS'],
            'features_to_download': ['TCS.NS', 'INFY.NS', 'WIPRO.NS', 'HCLTECH.NS', '^CNXIT', '^NSEI'],
            'rename_map': {'^CNXIT': 'Nifty_IT_Index', '^NSEI': 'Nifty_50_Index'},
            'model_path': 'models/best_model_it.keras', 'scaler_path': 'scalers/best_scaler_it.save'
        },
        {
            'sector_name': 'Auto', 'companies': ['TATAMOTORS.NS', 'MARUTI.NS', 'M&M.NS', 'BAJAJ-AUTO.NS'],
            'features_to_download': ['TATAMOTORS.NS', 'MARUTI.NS', 'M&M.NS', 'BAJAJ-AUTO.NS', '^CNXAUTO'],
            'rename_map': {'^CNXAUTO': 'Nifty_Auto_Index'},
            'model_path': 'models/best_model_auto.keras', 'scaler_path': 'scalers/best_scaler_auto.save'
        },
        {
            'sector_name': 'Banking', 'companies': ['HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS', 'KOTAKBANK.NS'],
            'features_to_download': ['HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS', 'KOTAKBANK.NS', '^NSEBANK'],
            'rename_map': {'^NSEBANK': 'Nifty_Bank_Index'},
            'model_path': 'models/best_model_banking.keras', 'scaler_path': 'scalers/best_scaler_banking.save'
        },
        {
            'sector_name': 'FMCG', 'companies': ['HINDUNILVR.NS', 'ITC.NS', 'NESTLEIND.NS', 'BRITANNIA.NS'],
            'features_to_download': ['HINDUNILVR.NS', 'ITC.NS', 'NESTLEIND.NS', 'BRITANNIA.NS', '^CNXFMCG'],
            'rename_map': {'^CNXFMCG': 'Nifty_FMCG_Index'},
            'model_path': 'models/best_model_fmcg.keras', 'scaler_path': 'scalers/best_scaler_fmcg.save'
        },
        {
            'sector_name': 'Pharma', 'companies': ['SUNPHARMA.NS', 'CIPLA.NS', 'DRREDDY.NS', 'DIVISLAB.NS'],
            'features_to_download': ['SUNPHARMA.NS', 'CIPLA.NS', 'DRREDDY.NS', 'DIVISLAB.NS', '^CNXPHARMA'],
            'rename_map': {'^CNXPHARMA': 'Nifty_Pharma_Index'},
            'model_path': 'models/best_model_pharma.keras', 'scaler_path': 'scalers/best_scaler_pharma.save'
        },
    ]
    for config in configs:
        model_path, scaler_path = config['model_path'], config['scaler_path']
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            try:
                # Use compile=False for faster loading when not retraining
                model = load_model(model_path, compile=False)
                scaler = joblib.load(scaler_path)
                
                # Dynamically determine feature and target columns
                training_columns = list(scaler.feature_names_in_)
                feature_cols = [col for col in training_columns if '_Return' not in col]
                target_cols = [col for col in training_columns if '_Return' in col]

                # Store all relevant info in the global dictionary
                SECTOR_MODELS[config['sector_name']] = {
                    'model': model,
                    'scaler': scaler,
                    'feature_cols': feature_cols,
                    'target_cols': target_cols,
                    'training_columns': training_columns,
                    **config
                }
                print(f"--- ✅ Loaded BEST model for {config['sector_name']} sector ---")
            except Exception as e:
                print(f"--- ❌ Failed loading model/scaler for {config['sector_name']}: {e} ---")
                traceback.print_exc()
        else:
            print(f"--- ⚠️ Model/scaler not found for {config['sector_name']} (paths: {model_path}, {scaler_path}) ---")

