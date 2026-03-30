from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import tensorflow.keras as keras
import os

app = Flask(__name__)
CORS(app)

# Load Models
MODEL_DIR = "."
print("Loading models and scaler...")
rf_model = joblib.load(os.path.join(MODEL_DIR, "rf_model.pkl"))
lstm_model = keras.models.load_model(os.path.join(MODEL_DIR, "lstm_model.keras"))
transformer_model = keras.models.load_model(os.path.join(MODEL_DIR, "transformer_model.keras"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
print("Models and scaler loaded successfully.")

# Load historical data context
# We need at least 11 previous months to create a sequence of 12 for the models,
# and potentially more to calculate the rolling windows and standardize them.
# SPEI-12 rolling sum requires 12 months. Standardization needs a longer history.
# For simplicity and correctness with the training data, we keep a global dataframe
# representing the history. In a real scenario, this would be a database.
DATA_FILE = "data/Kathmandu_Airport.csv"
if os.path.exists(DATA_FILE):
    historical_df = pd.read_csv(DATA_FILE)
    # Ensure date is string/datetime for appending later if needed, but we mainly care about WB and features
    historical_df['Date'] = pd.to_datetime(historical_df['Date'])
    print(f"Loaded {len(historical_df)} historical records from {DATA_FILE}.")
else:
    print(f"Warning: {DATA_FILE} not found. Starting with empty history.")
    historical_df = pd.DataFrame(columns=['Precipitation', 'Tmean', 'PET', 'WB', 'Date'])

def calculate_spei(df, window):
    rolling_sum = df['WB'].rolling(window=window).sum()
    mean = rolling_sum.mean()
    std = rolling_sum.std()
    spei = (rolling_sum - mean) / std
    return spei

@app.route('/predict', methods=['POST'])
def predict():
    global historical_df
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "No JSON payload provided"}), 400

        required_keys = ["precipitation", "tmean", "pet", "wb"]
        for k in required_keys:
            if k not in data:
                return jsonify({"status": "error", "message": f"Missing key: {k}"}), 400

        # Append new data point to historical dataframe
        new_row = {
            'Precipitation': float(data['precipitation']),
            'Tmean': float(data['tmean']),
            'PET': float(data['pet']),
            'WB': float(data['wb'])
        }

        # We might not have 'Date', just append with ignore_index
        historical_df = pd.concat([historical_df, pd.DataFrame([new_row])], ignore_index=True)

        # Calculate SPEI-3, 6, 12 dynamically over the updated history
        historical_df['SPEI-3'] = calculate_spei(historical_df, 3)
        historical_df['SPEI-6'] = calculate_spei(historical_df, 6)
        historical_df['SPEI-12'] = calculate_spei(historical_df, 12)

        # Drop NaN rows (mostly from rolling windows at the beginning)
        valid_df = historical_df.dropna(subset=['SPEI-3', 'SPEI-6', 'SPEI-12'])

        if len(valid_df) < 12:
            return jsonify({
                "status": "error",
                "message": "Not enough data points to form a sequence of length 12 after rolling calculations."
            }), 400

        features = ['Precipitation', 'Tmean', 'PET', 'SPEI-3', 'SPEI-6', 'SPEI-12']

        # Get the last 12 valid points
        seq_df = valid_df.tail(12)[features].copy()

        # Normalize using the loaded scaler
        seq_df[features] = scaler.transform(seq_df[features])

        # Prepare sequences for models
        # Shape needs to be (1, time_steps, features) for LSTM and Transformer
        # Shape needs to be (1, time_steps * features) for RF
        X_seq = seq_df.values.reshape(1, 12, len(features))
        X_flat = seq_df.values.reshape(1, -1)

        # Ensemble Inference
        transformer_pred = float(transformer_model.predict(X_seq, verbose=0).flatten()[0])
        lstm_pred = float(lstm_model.predict(X_seq, verbose=0).flatten()[0])
        rf_pred = float(rf_model.predict(X_flat)[0])

        response = {
            "status": "success",
            "predictions": {
                "transformer": round(transformer_pred, 2),
                "lstm": round(lstm_pred, 2),
                "random_forest": round(rf_pred, 2)
            },
            "metadata": {
                "target": "SPEI1_lead30",
                "unit": "Standardized Index"
            }
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)
