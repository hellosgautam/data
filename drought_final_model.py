import pandas as pd
import numpy as np
import os
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import tensorflow as pd_tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout, MultiHeadAttention, LayerNormalization, Flatten, GlobalAveragePooling1D

# Set random seed for reproducibility
np.random.seed(42)
pd_tf.random.set_seed(42)

def calculate_spei(df, window):
    rolling_sum = df['WB'].rolling(window=window).sum()
    # Simple standardization (z-score) as requested
    # Note: Traditional SPEI uses a distribution fitting (e.g., Log-Logistic),
    # but the instructions specify: "standardize these sums (z-score)"
    mean = rolling_sum.mean()
    std = rolling_sum.std()
    spei = (rolling_sum - mean) / std
    return spei

def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

def load_and_process_data(data_dir='data'):
    all_files = glob.glob(os.path.join(data_dir, "*.csv"))

    processed_dfs = []

    for filename in all_files:
        station_name = os.path.splitext(os.path.basename(filename))[0]
        df = pd.read_csv(filename)

        # Calculate SPEI-3, SPEI-6, SPEI-12
        df['SPEI-3'] = calculate_spei(df, 3)
        df['SPEI-6'] = calculate_spei(df, 6)
        df['SPEI-12'] = calculate_spei(df, 12)

        # Add Station name for identification
        df['Station'] = station_name

        # Drop NaNs created by rolling windows
        df = df.dropna()

        processed_dfs.append(df)

    return processed_dfs

def split_data(station_dfs, features, target, window_size=12):
    train_X_list, test_X_list = [], []
    train_y_list, test_y_list = [], []

    # For visualization
    kathmandu_test_X = None
    kathmandu_test_y = None
    kathmandu_dates = None # To plot against time if needed, though plot vs actual is requested

    for df in station_dfs:
        station_name = df['Station'].iloc[0]

        # Sort by date if 'Date' column exists, assuming it's already sorted based on file preview
        # df = df.sort_values('Date')

        X = df[features]
        y = df[target]

        # Train/Test Split (80/20) - Time-based
        split_idx = int(len(df) * 0.8)

        # We need to be careful with windowing.
        # If we window first, we lose some data at the beginning.
        # If we split first, we might have data leakage or discontinuity at the split.
        # Standard approach: Split time series, then window.
        # However, to maximize data usage for each station:

        # Create sequences
        X_seq, y_seq = create_dataset(X, y, window_size)

        # Split the sequences
        split_idx_seq = int(len(X_seq) * 0.8)

        X_train, X_test = X_seq[:split_idx_seq], X_seq[split_idx_seq:]
        y_train, y_test = y_seq[:split_idx_seq], y_seq[split_idx_seq:]

        train_X_list.append(X_train)
        test_X_list.append(X_test)
        train_y_list.append(y_train)
        test_y_list.append(y_test)

        if 'Kathmandu' in station_name: # Matches 'Kathmandu_Airport'
            kathmandu_test_X = X_test
            kathmandu_test_y = y_test


    # Concatenate all stations
    X_train_global = np.concatenate(train_X_list, axis=0)
    X_test_global = np.concatenate(test_X_list, axis=0)
    y_train_global = np.concatenate(train_y_list, axis=0)
    y_test_global = np.concatenate(test_y_list, axis=0)

    return X_train_global, X_test_global, y_train_global, y_test_global, kathmandu_test_X, kathmandu_test_y

# --- Models ---

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def build_transformer_model(input_shape):
    inputs = Input(shape=input_shape)

    # Multi-Head Attention
    # key_dim is the size of each attention head for query and key
    x = MultiHeadAttention(num_heads=4, key_dim=input_shape[-1])(inputs, inputs)
    x = LayerNormalization(epsilon=1e-6)(x + inputs) # Add & Norm

    # Feed Forward Part
    x = GlobalAveragePooling1D()(x) # Flatten the sequence
    x = Dropout(0.1)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.1)(x)
    outputs = Dense(1)(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

def main():
    # 1. Load and Process Data
    processed_dfs = load_and_process_data()

    features = ['Precipitation', 'Tmean', 'PET', 'SPEI-3', 'SPEI-6', 'SPEI-12']
    target = 'SPEI1_lead30'
    window_size = 12

    # Normalize features BEFORE splitting/windowing to ensure consistency across stations?
    # Or normalize after?
    # Best practice: Fit scaler on training data only.
    # Since we are doing a global model, we should probably fit on the global training set.
    # However, for simplicity in this workflow and since stations might have different scales (though standardized SPEI helps),
    # let's normalize per station or globally.
    # Given the prompt asks to "Normalize features", let's do it globally on the training set to avoid look-ahead bias,
    # but extracting the data first requires handling the 3D structure later.
    # A simpler approach for the 'load_and_process_data' step is to just return DFs, and we stack them to fit scaler.

    # Let's refine the workflow:
    # 1. Stack all data to fit scaler (on train split only ideally, but for simplicity let's do it on all non-test data or just be careful).
    # To strictly follow "Train/Test (80/20) per station", we should fit scaler on the 80% of each station concatenated.

    # Re-implementing normalization logic correctly:
    all_data_train_list = []
    for df in processed_dfs:
        split_idx = int(len(df) * 0.8)
        all_data_train_list.append(df.iloc[:split_idx][features])

    global_train_df = pd.concat(all_data_train_list)
    scaler = StandardScaler()
    scaler.fit(global_train_df)

    # Apply transform to all dfs
    valid_dfs = []
    for df in processed_dfs:
        if len(df) > 0:
            df[features] = scaler.transform(df[features])
            valid_dfs.append(df)
        else:
            print(f"Warning: Station {df['Station'].iloc[0] if not df.empty else 'Unknown'} has 0 samples after processing.")

    processed_dfs = valid_dfs

    # 2. Split Data
    X_train, X_test, y_train, y_test, kath_X, kath_y = split_data(processed_dfs, features, target, window_size)

    print(f"Training Data Shape: {X_train.shape}")
    print(f"Test Data Shape: {X_test.shape}")

    # 3. Model A: Random Forest
    # RF needs 2D input (samples, features*window) or just (samples, features) if not using window.
    # The prompt implies comparing models, and usually RF uses the same features.
    # If we use the windowed data, we flatten it.
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    kath_X_flat = kath_X.reshape(kath_X.shape[0], -1)

    print("Training Random Forest...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_flat, y_train)
    y_pred_rf = rf_model.predict(X_test_flat)

    # 4. Model B: LSTM
    print("Training LSTM...")
    lstm_model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    lstm_model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1, verbose=1)
    y_pred_lstm = lstm_model.predict(X_test).flatten()

    # 5. Model C: Transformer
    print("Training Transformer...")
    transformer_model = build_transformer_model((X_train.shape[1], X_train.shape[2]))
    transformer_model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1, verbose=1)
    y_pred_transformer = transformer_model.predict(X_test).flatten()

    # 6. Evaluation
    results = []

    models = {
        'Random Forest': y_pred_rf,
        'LSTM': y_pred_lstm,
        'Transformer': y_pred_transformer
    }

    for name, y_pred in models.items():
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        results.append({'Model': name, 'R2': r2, 'MAE': mae, 'RMSE': rmse})

    results_df = pd.DataFrame(results)
    print("\nResults Summary:")
    print(results_df)
    results_df.to_csv('results_summary.csv', index=False)

    # 7. Visualization (Kathmandu Airport)
    # Predict for Kathmandu
    kath_pred_rf = rf_model.predict(kath_X_flat)
    kath_pred_lstm = lstm_model.predict(kath_X).flatten()
    kath_pred_transformer = transformer_model.predict(kath_X).flatten()

    plt.figure(figsize=(12, 6))
    plt.plot(kath_y, label='Actual', color='black', linewidth=2)
    plt.plot(kath_pred_rf, label='Random Forest', linestyle='--')
    plt.plot(kath_pred_lstm, label='LSTM', linestyle='--')
    plt.plot(kath_pred_transformer, label='Transformer', linestyle='--')

    plt.title('Drought Forecasting: Predicted vs Actual (Kathmandu Airport)')
    plt.xlabel('Time Steps (Months)')
    plt.ylabel('SPEI1_lead30')
    plt.legend()
    plt.grid(True)
    plt.savefig('kathmandu_comparison.png')
    print("Visualization saved as kathmandu_comparison.png")

if __name__ == "__main__":
    main()
