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
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)
pd_tf.random.set_seed(42)

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

    spei_windows = [1, 6, 9, 12]
    lead_times = [30, 60, 90]

    for filename in all_files:
        station_name = os.path.splitext(os.path.basename(filename))[0]
        df = pd.read_csv(filename)
        df['Station'] = station_name

        # Ensure only fully populated rows are kept.
        cols_to_check = ['Precipitation', 'Tmean', 'PET']
        for window in spei_windows:
            cols_to_check.append(f'SPEI-{window}')
            for lead in lead_times:
                cols_to_check.append(f'SPEI-{window}_lead{lead}')

        # Only keep rows where all these columns have values
        available_cols = [c for c in cols_to_check if c in df.columns]
        df = df.dropna(subset=available_cols)

        if len(df) > 0:
            processed_dfs.append(df)
        else:
            print(f"Warning: Station {station_name} has 0 samples after processing. Skipping.")

    return processed_dfs

def evaluate_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    # RRMSE
    mean_true = np.mean(y_true)
    if mean_true != 0:
        rrmse = (rmse / np.abs(mean_true)) * 100
    else:
        rrmse = np.nan

    return r2, mae, mse, rmse, rrmse

def split_and_scale(processed_dfs, feature_cols, target_col, window_size=12):
    # To keep track of each station's test data separately for final plotting
    station_test_data = {}

    # Fit scaler globally
    all_train_features = []
    for df in processed_dfs:
        split_idx = int(len(df) * 0.8)
        all_train_features.append(df.iloc[:split_idx][feature_cols])

    global_train_df = pd.concat(all_train_features)
    scaler = StandardScaler()
    scaler.fit(global_train_df)

    train_X_list, test_X_list = [], []
    train_y_list, test_y_list = [], []

    for df in processed_dfs:
        station_name = df['Station'].iloc[0]
        df_scaled = df.copy()
        df_scaled[feature_cols] = scaler.transform(df_scaled[feature_cols])

        X = df_scaled[feature_cols]
        y = df_scaled[target_col]

        X_seq, y_seq = create_dataset(X, y, window_size)

        split_idx_seq = int(len(X_seq) * 0.8)

        X_train, X_test = X_seq[:split_idx_seq], X_seq[split_idx_seq:]
        y_train, y_test = y_seq[:split_idx_seq], y_seq[split_idx_seq:]

        train_X_list.append(X_train)
        test_X_list.append(X_test)
        train_y_list.append(y_train)
        test_y_list.append(y_test)

        station_test_data[station_name] = {
            'X_test': X_test,
            'y_test': y_test
        }

    X_train_global = np.concatenate(train_X_list, axis=0)
    X_test_global = np.concatenate(test_X_list, axis=0)
    y_train_global = np.concatenate(train_y_list, axis=0)
    y_test_global = np.concatenate(test_y_list, axis=0)

    return X_train_global, X_test_global, y_train_global, y_test_global, station_test_data

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def build_transformer_model(input_shape):
    inputs = Input(shape=input_shape)
    x = MultiHeadAttention(num_heads=2, key_dim=16)(inputs, inputs)
    x = LayerNormalization(epsilon=1e-6)(x + inputs)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.1)(x)
    x = Dense(32, activation="relu")(x)
    outputs = Dense(1)(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

def main():
    os.makedirs('output', exist_ok=True)
    os.makedirs('output/plots', exist_ok=True)

    processed_dfs = load_and_process_data()

    spei_windows = [1, 6, 9, 12]
    lead_times = [30, 60, 90]

    window_size = 6 # Window size for sequences

    all_results = []

    # Process each target independently
    for spei_win in spei_windows:
        for lead_time in lead_times:
            target_col = f'SPEI-{spei_win}_lead{lead_time}'
            print(f"\n====================== Training for Target: {target_col} ======================")

            # The features will be Precipitation, Tmean, PET, and all the available calculated SPEIs
            feature_cols = ['Precipitation', 'Tmean', 'PET'] + [f'SPEI-{w}' for w in spei_windows]

            # Since some merged files might not have all columns correctly named,
            # Let's verify feature_cols and target_col exists in the first processed_df
            if target_col not in processed_dfs[0].columns:
                print(f"Target column '{target_col}' not found. Skipping.")
                continue

            available_feature_cols = [c for c in feature_cols if c in processed_dfs[0].columns]

            X_train, X_test, y_train, y_test, station_test_data = split_and_scale(processed_dfs, available_feature_cols, target_col, window_size)

            if len(X_train) == 0:
                print(f"Not enough data to train for {target_col}.")
                continue

            # 1. Random Forest
            print(f"Training Random Forest...")
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
            rf_model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
            rf_model.fit(X_train_flat, y_train)

            # 2. LSTM
            print(f"Training LSTM...")
            lstm_model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
            lstm_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0, validation_split=0.1)

            # 3. Transformer
            print(f"Training Transformer...")
            transformer_model = build_transformer_model((X_train.shape[1], X_train.shape[2]))
            transformer_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0, validation_split=0.1)

            # Station-level Evaluation and Plotting
            for station, data in station_test_data.items():
                s_X_test = data['X_test']
                s_y_test = data['y_test']

                if len(s_y_test) == 0:
                    continue

                s_X_test_flat = s_X_test.reshape(s_X_test.shape[0], -1)

                # Predictions
                y_pred_rf = rf_model.predict(s_X_test_flat)
                y_pred_lstm = lstm_model.predict(s_X_test, verbose=0).flatten()
                y_pred_transformer = transformer_model.predict(s_X_test, verbose=0).flatten()

                models = {
                    'Random Forest': y_pred_rf,
                    'LSTM': y_pred_lstm,
                    'Transformer': y_pred_transformer
                }

                # Metrics Collection
                for model_name, preds in models.items():
                    r2, mae, mse, rmse, rrmse = evaluate_metrics(s_y_test, preds)
                    all_results.append({
                        'Station': station,
                        'SPEI': f'SPEI-{spei_win}',
                        'Lead Time': f'Lead {lead_time}',
                        'Model': model_name,
                        'MAE': mae,
                        'MSE': mse,
                        'RMSE': rmse,
                        'RRMSE (%)': rrmse,
                        'R2': r2
                    })

                # Plotting (only predicted portion / test set)
                plt.figure(figsize=(10, 5))
                plt.plot(s_y_test, label='Actual', color='black', linewidth=2)
                plt.plot(y_pred_rf, label='Random Forest', linestyle='--')
                plt.plot(y_pred_lstm, label='LSTM', linestyle='-.')
                plt.plot(y_pred_transformer, label='Transformer', linestyle=':')

                plt.title(f'{station} - SPEI-{spei_win} Lead {lead_time}')
                plt.xlabel('Test Time Steps (Months)')
                plt.ylabel(f'SPEI-{spei_win} Lead {lead_time}')
                plt.legend()
                plt.grid(True)

                # Save plot
                target_plot_dir = os.path.join('output', 'plots', f'SPEI-{spei_win}_Lead_{lead_time}')
                os.makedirs(target_plot_dir, exist_ok=True)
                plot_filename = os.path.join(target_plot_dir, f'{station}.png')
                plt.savefig(plot_filename)
                plt.close()

    results_df = pd.DataFrame(all_results)
    results_df.to_csv('output/performance_parameters.csv', index=False)
    print("Done! Metrics saved to 'output/performance_parameters.csv' and plots saved to 'output/plots/' directory.")

if __name__ == "__main__":
    main()
