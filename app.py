from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)
CORS(app)

# Load the scaler and models
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

lstm_model = load_model('lstm_model.keras')
transformer_model = load_model('transformer_model.keras')

# Preload historical data for calculating SPEI and sequence features.
# Since the model uses sequences of 12 (window_size = 12), we need at least 12 months for the sequence.
# But for SPEI-12, the first 11 rolling sums are NaN. So to have 12 non-NaN months at the end,
# we need 12 + 11 = 23 months of history + 1 incoming = 24 months total.
try:
    history_df = pd.read_csv('data/Kathmandu_Airport.csv')
    # Keep the last 23 rows for history buffer
    history_df = history_df[['Precipitation', 'Tmean', 'PET', 'WB']].tail(23).reset_index(drop=True)
except Exception as e:
    print(f"Failed to load history data: {e}")
    # Fallback to dummy data
    history_df = pd.DataFrame({
        'Precipitation': [0.0]*23,
        'Tmean': [0.0]*23,
        'PET': [0.0]*23,
        'WB': [0.0]*23
    })

def calculate_spei(df, window):
    rolling_sum = df['WB'].rolling(window=window).sum()
    mean = rolling_sum.mean()
    std = rolling_sum.std()

    if std == 0 or pd.isna(std):
        return pd.Series([0.0] * len(df), index=df.index)

    spei = (rolling_sum - mean) / std
    return spei

@app.route('/predict', methods=['POST'])
def predict():
    try:
        global history_df
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        # Required fields
        for k in ['precipitation', 'tmean', 'pet', 'wb']:
            if k not in data:
                return jsonify({"error": f"Missing field: {k}"}), 400

        new_row = pd.DataFrame({
            'Precipitation': [data['precipitation']],
            'Tmean': [data['tmean']],
            'PET': [data['pet']],
            'WB': [data['wb']]
        })

        # Append new row
        current_df = pd.concat([history_df, new_row], ignore_index=True)

        # Calculate SPEI features on the whole buffer
        current_df['SPEI-3'] = calculate_spei(current_df, 3)
        current_df['SPEI-6'] = calculate_spei(current_df, 6)
        current_df['SPEI-12'] = calculate_spei(current_df, 12)

        # Get the last 12 months (window_size for LSTM/Transformer)
        features = ['Precipitation', 'Tmean', 'PET', 'SPEI-3', 'SPEI-6', 'SPEI-12']
        recent_12 = current_df[features].tail(12).copy()

        # Update history for next request (keep last 23)
        history_df = current_df[['Precipitation', 'Tmean', 'PET', 'WB']].tail(23).reset_index(drop=True)

        # Apply standard scaler
        scaled_features = scaler.transform(recent_12)

        # Reshape for sequence models (LSTM, Transformer): (samples, time_steps, features)
        X_seq = np.array([scaled_features])

        # Reshape for Random Forest: flattened (samples, time_steps * features)
        X_flat = X_seq.reshape(1, -1)

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
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
