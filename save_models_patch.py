import pandas as pd
import numpy as np
import os
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import tensorflow as pd_tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout, MultiHeadAttention, LayerNormalization, Flatten, GlobalAveragePooling1D
import pickle

from drought_final_model import load_and_process_data, split_data, build_lstm_model, build_transformer_model

def main():
    processed_dfs = load_and_process_data()
    features = ['Precipitation', 'Tmean', 'PET', 'SPEI-3', 'SPEI-6', 'SPEI-12']
    target = 'SPEI1_lead30'
    window_size = 12

    all_data_train_list = []
    for df in processed_dfs:
        split_idx = int(len(df) * 0.8)
        all_data_train_list.append(df.iloc[:split_idx][features])

    global_train_df = pd.concat(all_data_train_list)
    scaler = StandardScaler()
    scaler.fit(global_train_df)

    valid_dfs = []
    for df in processed_dfs:
        if len(df) > 0:
            df[features] = scaler.transform(df[features])
            valid_dfs.append(df)

    processed_dfs = valid_dfs

    X_train, X_test, y_train, y_test, kath_X, kath_y = split_data(processed_dfs, features, target, window_size)

    X_train_flat = X_train.reshape(X_train.shape[0], -1)

    print("Training Random Forest...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_flat, y_train)

    print("Training LSTM...")
    lstm_model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    lstm_model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1, verbose=1)

    print("Training Transformer...")
    transformer_model = build_transformer_model((X_train.shape[1], X_train.shape[2]))
    transformer_model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1, verbose=1)

    # Save models
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('rf_model.pkl', 'wb') as f:
        pickle.dump(rf_model, f)
    lstm_model.save('lstm_model.keras')
    transformer_model.save('transformer_model.keras')
    print("Models saved successfully!")

if __name__ == "__main__":
    main()
