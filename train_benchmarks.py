"""
Hierarchical vs. Sequential Processing for Earthquake Forecasting
Author: Marzieh Khalili
Affiliation: Shiraz University, Iran
Date: 2025
Description: 
    This script benchmarks TCN, Transformer, LSTM, and GRU architectures 
    for spatiotemporal seismic forecasting in the Zagros-Makran transition zone.
    It implements strict chronological splitting to prevent data leakage.
"""

import os
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (LSTM, Dense, Dropout, GRU, MultiHeadAttention,
                                     LayerNormalization, Input, GlobalAveragePooling1D,
                                     Add, Embedding, Conv1D)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K

# ==========================================
# 1. REPRODUCIBILITY SETUP
# ==========================================
def set_seeds(seed=42):
    """
    Sets random seeds for reproducibility across Python, Numpy, and TensorFlow.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

set_seeds()
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# ==========================================
# 2. DATA PREPARATION UTILITIES
# ==========================================
def create_sequences(features, targets, window_size, horizon):
    """
    Creates sliding window sequences for time-series forecasting.
    
    Args:
        features (np.array): Input features matrix (N, Features).
        targets (np.array): Target variables matrix (N, Targets).
        window_size (int): Look-back window size (w).
        horizon (int): Forecasting horizon (h).
        
    Returns:
        tuple: (X, Y) arrays suitable for Keras models.
    """
    X, Y = [], []
    n_len = len(features)
    for i in range(n_len - window_size - horizon + 1):
        seq_x = features[i : i + window_size]
        seq_y = targets[i + window_size : i + window_size + horizon]
        X.append(seq_x)
        Y.append(seq_y.flatten())
    return np.array(X), np.array(Y)

def get_chronological_splits(df, train_ratio=0.8, val_ratio=0.1):
    """
    Splits the dataset chronologically to prevent future-to-past data leakage.
    
    Args:
        df (pd.DataFrame): Sorted dataframe containing the seismic catalog.
        
    Returns:
        tuple: (train_df, val_df, test_df)
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    return train_df, val_df, test_df

# ==========================================
# 3. MODEL ARCHITECTURES
# ==========================================
def build_tcn_native(window_size, n_features, out_dim):
    """
    Constructs a Temporal Convolutional Network (TCN) with dilated causal convolutions.
    Dilations: [1, 2, 4, 8, 16] to expand receptive field.
    """
    inputs = Input(shape=(window_size, n_features))
    x = inputs
    dilations = [1, 2, 4, 8, 16]
    
    for d in dilations:
        # Causal padding ensures no leakage from future steps
        conv = Conv1D(filters=64, kernel_size=3, dilation_rate=d, padding='causal', activation='relu')(x)
        if x.shape[-1] != 64:
            x = Conv1D(filters=64, kernel_size=1)(x)
        x = Add()([x, conv])
        x = Dropout(0.2)(x)
    
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(out_dim)(x)
    return Model(inputs, outputs, name="TCN")

def build_transformer(window_size, n_features, out_dim):
    """
    Constructs a Transformer Encoder with Multi-Head Attention and Positional Embeddings.
    """
    inputs = Input(shape=(window_size, n_features))
    
    d_model = 64
    x = Dense(d_model)(inputs)
    
    # Static Positional Encoding
    positions = tf.range(start=0, limit=window_size, delta=1)
    pos_emb = Embedding(input_dim=window_size, output_dim=d_model, trainable=False)(positions)
    x = Add()([x, tf.expand_dims(pos_emb, axis=0)])
    
    # Self-Attention Block
    att = MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
    x = Add()([x, att])
    x = LayerNormalization(epsilon=1e-6)(x)
    
    # Feed-Forward Network
    ffn = Sequential([Dense(d_model, activation="relu"), Dense(d_model)])(x)
    x = Add()([x, ffn])
    x = LayerNormalization(epsilon=1e-6)(x)
    
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation="relu")(x)
    outputs = Dense(out_dim)(x)
    
    return Model(inputs, outputs, name="Transformer")

def get_model(model_name, w_size, n_features, out_dim):
    """
    Factory method to instantiate models based on architecture name.
    """
    if model_name == "LSTM":
        return Sequential([
            Input(shape=(w_size, n_features)),
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dense(out_dim)
        ], name="LSTM")
        
    elif model_name == "GRU":
        return Sequential([
            Input(shape=(w_size, n_features)),
            GRU(64, return_sequences=False),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dense(out_dim)
        ], name="GRU")
    
    elif model_name == "TCN":
        return build_tcn_native(w_size, n_features, out_dim)
        
    elif model_name == "Transformer":
        return build_transformer(w_size, n_features, out_dim)
        
    # Heavy Variants (Parameter-matched for w=50)
    elif model_name == "LSTM_Heavy":
        return Sequential([
            Input(shape=(w_size, n_features)),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dense(out_dim)
        ], name="LSTM_Heavy")
        
    elif model_name == "GRU_Heavy":
        return Sequential([
            Input(shape=(w_size, n_features)),
            GRU(64, return_sequences=True),
            Dropout(0.2),
            GRU(64, return_sequences=False),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dense(out_dim)
        ], name="GRU_Heavy")
    
    else:
        raise ValueError(f"Unknown model: {model_name}")

def plot_loss_curves(history, title, filename):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Val Loss', linestyle='--', linewidth=2)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

# ==========================================
# 4. MAIN EXPERIMENT LOOP
# ==========================================
def main():
    DATA_FILE = 'Final_Cleaned_Catalog_v2.csv'
    CUTOFF_YEAR = 1998
    HORIZON = 10
    EPOCHS = 50
    BATCH_SIZE = 16
    
    print("--- Phase 1: Data Loading & Preprocessing ---")
    try:
        df = pd.read_csv(DATA_FILE)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df[df['datetime'].dt.year >= CUTOFF_YEAR].reset_index(drop=True)
    except Exception as e:
        print(f"Dataset error: {e}")
        # Dummy data generation for testing purposes if file is missing
        print("Generating DUMMY data for demonstration...")
        dates = pd.date_range(start='1998-01-01', periods=3000, freq='D')
        df = pd.DataFrame({
            'datetime': dates,
            'LAT': np.random.uniform(25, 29, 3000),
            'LON': np.random.uniform(54, 58, 3000),
            'DEPTH': np.random.uniform(5, 30, 3000),
            'MAG': np.random.exponential(scale=1.0, size=3000) + 2.85,
            'time_diff': np.random.exponential(scale=10000, size=3000)
        })

    # Log-transform inter-event time to handle skewness
    df['time_diff_log'] = np.log1p(df['time_diff'])
    
    feature_cols = ['LAT', 'LON', 'DEPTH', 'MAG', 'time_diff_log']
    target_cols = ['time_diff', 'LAT', 'LON', 'MAG'] 

    # 1. Split Data
    train_df, val_df, test_df = get_chronological_splits(df)

    # 2. Scale Data (Fit on Train ONLY)
    scaler_x = MinMaxScaler().fit(train_df[feature_cols])
    scaler_y = MinMaxScaler().fit(train_df[target_cols])

    def get_scaled_data(dframe):
        return scaler_x.transform(dframe[feature_cols]), scaler_y.transform(dframe[target_cols])

    x_train_s, y_train_s = get_scaled_data(train_df)
    x_val_s, y_val_s = get_scaled_data(val_df)
    x_test_s, y_test_s = get_scaled_data(test_df)

    # Loop over window sizes (Short-term vs Medium-term)
    for w_size in [20, 50]:
        print(f"\n{'#'*50}")
        print(f"### RUNNING EXPERIMENT: WINDOW SIZE {w_size}")
        print(f"{'#'*50}")
        
        X_train, Y_train = create_sequences(x_train_s, y_train_s, w_size, HORIZON)
        X_val, Y_val = create_sequences(x_val_s, y_val_s, w_size, HORIZON)
        X_test, Y_test = create_sequences(x_test_s, y_test_s, w_size, HORIZON)
        
        n_features = X_train.shape[2]
        out_dim = Y_train.shape[1]
        
        models_to_run = ["LSTM", "GRU", "TCN", "Transformer"]
        if w_size == 50:
            models_to_run += ["LSTM_Heavy", "GRU_Heavy"]
            
        results_list = []
        
        for m_name in models_to_run:
            print(f"--> Training Model: {m_name}...")
            
            K.clear_session()
            
            model = get_model(m_name, w_size, n_features, out_dim)
            model.compile(optimizer='adam', loss='mse')
            
            es = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
            history = model.fit(
                X_train, Y_train,
                validation_data=(X_val, Y_val),
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                callbacks=[es],
                verbose=0
            )
            
            plot_name = f"Loss_{m_name}_w{w_size}.png"
            plot_loss_curves(history, f"{m_name} (w={w_size})", plot_name)
            
            # Evaluation
            Y_pred = model.predict(X_test, verbose=0)
            
            # Reshape for metric calculation (Samples, Horizon, Features)
            Y_test_3d = Y_test.reshape(-1, HORIZON, len(target_cols))
            Y_pred_3d = Y_pred.reshape(-1, HORIZON, len(target_cols))
            
            # Metric: Magnitude MAE (Index 3 is Magnitude)
            mag_mae = mean_absolute_error(Y_test_3d[:,:,3], Y_pred_3d[:,:,3])
            overall_mae = mean_absolute_error(Y_test, Y_pred)
            
            # Sample-wise error for statistical testing
            sample_errors = np.mean(np.abs(Y_test_3d[:,:,3] - Y_pred_3d[:,:,3]), axis=1)
            
            results_list.append({
                "Model": m_name,
                "Magnitude MAE": mag_mae,
                "Overall MAE": overall_mae,
                "Params": model.count_params(),
                "Errors": sample_errors
            })
            
            print(f"    Done. Mag MAE: {mag_mae:.5f}")

        # Results Summary
        res_df = pd.DataFrame(results_list).sort_values("Magnitude MAE")
        
        print(f"\n>>> FINAL LEADERBOARD (Window {w_size}) <<<")
        print("-" * 65)
        print(f"{'Model':<15} | {'Mag MAE':<10} | {'Overall MAE':<12} | {'Params':<10}")
        print("-" * 65)
        for _, row in res_df.iterrows():
            print(f"{row['Model']:<15} | {row['Magnitude MAE']:.5f}    | {row['Overall MAE']:.5f}      | {row['Params']:<10}")
        print("-" * 65)
        
        # Paired T-Test
        winner = res_df.iloc[0]
        print(f"\n WINNER: {winner['Model']}")
        
        if len(res_df) > 1:
            print(f"Statistical Significance Test (vs {winner['Model']}):")
            for i in range(1, len(res_df)):
                competitor = res_df.iloc[i]
                t_stat, p_val = stats.ttest_rel(winner['Errors'], competitor['Errors'])
                
                sig_label = "Not Significant"
                if p_val < 0.05: sig_label = "Significant (*)"
                if p_val < 0.001: sig_label = "Highly Significant (***)"
                
                print(f"   vs {competitor['Model']:<12}: p-value = {p_val:.2e} -> {sig_label}")
        print("\n")

if __name__ == "__main__":
    main()