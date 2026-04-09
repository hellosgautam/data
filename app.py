from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__,
            template_folder=os.path.join(BASE_DIR, 'templates'),
            static_folder=os.path.join(BASE_DIR, 'static'))
CORS(app)

# Load the scaler and models
with open(os.path.join(BASE_DIR, 'scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)
with open(os.path.join(BASE_DIR, 'rf_model.pkl'), 'rb') as f:
    rf_model = pickle.load(f)

lstm_model = load_model(os.path.join(BASE_DIR, 'lstm_model.keras'))
transformer_model = load_model(os.path.join(BASE_DIR, 'transformer_model.keras'))

try:
    history_df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'Kathmandu_Airport.csv'))
    history_df = history_df[['Precipitation', 'Tmean', 'PET', 'WB']].tail(23).reset_index(drop=True)
except Exception as e:
    print(f"Failed to load history data: {e}")
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

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        global history_df
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        for k in ['precipitation', 'tmean', 'pet', 'wb']:
            if k not in data:
                return jsonify({"error": f"Missing field: {k}"}), 400

        new_row = pd.DataFrame({
            'Precipitation': [data['precipitation']],
            'Tmean': [data['tmean']],
            'PET': [data['pet']],
            'WB': [data['wb']]
        })

        current_df = pd.concat([history_df, new_row], ignore_index=True)

        current_df['SPEI-3'] = calculate_spei(current_df, 3)
        current_df['SPEI-6'] = calculate_spei(current_df, 6)
        current_df['SPEI-12'] = calculate_spei(current_df, 12)

        features = ['Precipitation', 'Tmean', 'PET', 'SPEI-3', 'SPEI-6', 'SPEI-12']
        recent_12 = current_df[features].tail(12).copy()

        history_df = current_df[['Precipitation', 'Tmean', 'PET', 'WB']].tail(23).reset_index(drop=True)

        scaled_features = scaler.transform(recent_12)
        X_seq = np.array([scaled_features])
        X_flat = X_seq.reshape(1, -1)

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
