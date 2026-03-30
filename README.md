# Drought Forecasting and Monitoring using Machine Learning

This is an end-to-end Machine Learning pipeline and API for forecasting drought conditions (SPEI1_lead30) using a multi-model ensemble approach (Transformer, LSTM, Random Forest).

## Project Setup

1. **Install Dependencies:**
   Install the required Python packages using pip:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Flask Backend & UI:**
   Start the local web server:
   ```bash
   python3 app.py
   ```

   Once running, you can access the frontend dashboard in your browser at:
   `http://127.0.0.1:5000/`

3. **API Documentation:**
   The backend also serves a direct API endpoint at `/predict`.
   - **Method:** `POST`
   - **Payload:** JSON format containing `precipitation`, `tmean`, `pet`, and `wb`.
   - **Response:** JSON format with predictions from the `transformer`, `lstm`, and `random_forest` models.

## Model Architecture Transparency
This project utilizes a powerful ensemble of deep learning and traditional machine learning algorithms to capture non-linear temporal dynamics in weather data. The input data undergoes automated feature engineering (rolling SPEI calculation and Z-score standardization) before inference.
