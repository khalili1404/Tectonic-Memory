"""
SHAP Analysis for Seismic Forecasting Models
Author: Marzieh Khalili
Description:
    This script performs post-hoc interpretability analysis using SHAP (DeepExplainer/KernelExplainer)
    to quantify feature importance and temporal attention mechanisms of trained deep learning models.
"""

import os
import warnings
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model, Sequential, Model
from tensorflow.keras.layers import (LSTM, Dense, Dropout, GRU, MultiHeadAttention,
                                     LayerNormalization, Input, GlobalAveragePooling1D,
                                     Add, Embedding, Conv1D)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K

# ==========================================
# 1. REPRODUCIBILITY SETUP
# ==========================================
def set_seeds(seed=42):
    """Sets random seeds for reproducibility."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

set_seeds()
warnings.filterwarnings('ignore')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

DATA_FILE = 'Final_Cleaned_Catalog_v2.csv'
CUTOFF_YEAR = 1998
HORIZON = 10
EPOCHS = 35
BATCH_SIZE = 16
TARGET_FEAT_IDX = 3  # Index 3 corresponds to 'Magnitude'

FEATURE_NAMES = ['Latitude', 'Longitude', 'Depth', 'Magnitude', 'Inter-event Time (Log)']

# ==========================================
# 2. MODEL FACTORY
# ==========================================
def build_model_factory(model_name, w_size, n_features, out_dim):
    """Reconstructs model architecture for on-the-fly retraining if saved files are missing."""
    if model_name == "LSTM":
        return Sequential([
            Input(shape=(w_size, n_features)),
            LSTM(64, return_sequences=False), Dropout(0.2),
            Dense(64, activation='relu'), Dense(out_dim)
        ])
    elif model_name == "GRU":
        return Sequential([
            Input(shape=(w_size, n_features)),
            GRU(64, return_sequences=False), Dropout(0.2),
            Dense(64, activation='relu'), Dense(out_dim)
        ])
    elif model_name == "TCN":
        inputs = Input(shape=(w_size, n_features))
        x = inputs
        # Dilations [1, 2, 4, 8, 16] ensure receptive field covers the window
        for d in [1, 2, 4, 8, 16]:
            conv = Conv1D(64, 3, dilation_rate=d, padding='causal', activation='relu')(x)
            if x.shape[-1] != 64:
                x = Conv1D(64, 1)(x)
            x = Add()([x, conv])
            x = Dropout(0.2)(x)
        x = GlobalAveragePooling1D()(x)
        x = Dense(64, activation='relu')(x)
        return Model(inputs, Dense(out_dim)(x))
    elif model_name == "Transformer":
        inputs = Input(shape=(w_size, n_features))
        d_model = 64
        x = Dense(d_model)(inputs)
        pos = tf.range(start=0, limit=w_size, delta=1)
        emb = Embedding(w_size, d_model, trainable=False)(pos)
        x = Add()([x, tf.expand_dims(emb, axis=0)])
        att = MultiHeadAttention(4, 16)(x, x)
        x = LayerNormalization(epsilon=1e-6)(Add()([x, att]))
        ffn = Sequential([Dense(d_model, "relu"), Dense(d_model)])(x)
        x = LayerNormalization(epsilon=1e-6)(Add()([x, ffn]))
        x = GlobalAveragePooling1D()(x)
        x = Dropout(0.2)(x)
        x = Dense(64, "relu")(x)
        return Model(inputs, Dense(out_dim)(x))
    elif model_name == "LSTM_Heavy":
        return Sequential([
            Input(shape=(w_size, n_features)),
            LSTM(64, return_sequences=True), Dropout(0.2),
            LSTM(64, return_sequences=False), Dropout(0.2),
            Dense(64, activation='relu'), Dense(out_dim)
        ])
    elif model_name == "GRU_Heavy":
        return Sequential([
            Input(shape=(w_size, n_features)),
            GRU(64, return_sequences=True), Dropout(0.2),
            GRU(64, return_sequences=False), Dropout(0.2),
            Dense(64, activation='relu'), Dense(out_dim)
        ])
    return None

# ==========================================
# 3. DATA PREPARATION
# ==========================================
def create_sequences(features, targets, window_size, horizon):
    """Generates sliding window sequences."""
    X, Y = [], []
    for i in range(len(features) - window_size - horizon + 1):
        X.append(features[i : i + window_size])
        Y.append(targets[i + window_size : i + window_size + horizon].flatten())
    return np.array(X), np.array(Y)

def get_chronological_splits(df):
    """Splits data chronologically (80/10/10) to prevent leakage."""
    n = len(df)
    train_end, val_end = int(n * 0.8), int(n * 0.9)
    return df.iloc[:train_end].copy(), df.iloc[train_end:val_end].copy(), df.iloc[val_end:].copy()

print("--- Data Loading ---")
try:
    df = pd.read_csv(DATA_FILE)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df[df['datetime'].dt.year >= CUTOFF_YEAR].reset_index(drop=True)
except:
    print("WARNING: Using Dummy Data")
    dates = pd.date_range('1998-01-01', periods=3000, freq='D')
    df = pd.DataFrame({
        'datetime': dates, 
        'LAT': np.random.uniform(25, 29, 3000), 
        'LON': np.random.uniform(54, 58, 3000),
        'DEPTH': np.random.uniform(5, 30, 3000), 
        'MAG': np.random.exponential(1, 3000) + 2.85, 
        'time_diff': np.random.exponential(10000, 3000)
    })

df['time_diff_log'] = np.log1p(df['time_diff'])
cols_x = ['LAT', 'LON', 'DEPTH', 'MAG', 'time_diff_log']
cols_y = ['time_diff', 'LAT', 'LON', 'MAG']

train_df, val_df, test_df = get_chronological_splits(df)
scaler_x = MinMaxScaler().fit(train_df[cols_x])
scaler_y = MinMaxScaler().fit(train_df[cols_y])

x_train_s, y_train_s = scaler_x.transform(train_df[cols_x]), scaler_y.transform(train_df[cols_y])
x_test_s, y_test_s = scaler_x.transform(test_df[cols_x]), scaler_y.transform(test_df[cols_y])
x_val_s, y_val_s = scaler_x.transform(val_df[cols_x]), scaler_y.transform(val_df[cols_y])

# ==========================================
# 4. SHAP ANALYSIS UTILITIES
# ==========================================
def get_or_train_model(name, w_size, X_tr, Y_tr, X_va, Y_va):
    """Loads a pre-trained model or retrains it if the file is missing."""
    filename = f"Model_{name}_w{w_size}.keras"
    if not os.path.exists(filename) and os.path.exists(f"model_{name}_w{w_size}.keras"):
        filename = f"model_{name}_w{w_size}.keras"
        
    try:
        model = load_model(filename)
        print(f"   Loaded: {filename}")
        return model
    except:
        print(f"    File {filename} not found. Retraining {name}...")
        K.clear_session()
        n_feat, out_dim = X_tr.shape[2], Y_tr.shape[1]
        model = build_model_factory(name, w_size, n_feat, out_dim)
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_tr, Y_tr, validation_data=(X_va, Y_va), 
                  epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0,
                  callbacks=[EarlyStopping(patience=5, restore_best_weights=True)])
        model.save(filename)
        print(f"   Training Complete & Saved: {filename}")
        return model

def analyze_model(name, w_size, X_tr, Y_tr, X_va, Y_va, X_te, Y_te):
    """Performs SHAP Kernel Explainer analysis and generates interpretation plots."""
    print(f"\n>> Processing: {name} (w={w_size})")
    
    model = get_or_train_model(name, w_size, X_tr, Y_tr, X_va, Y_va)
    
    # Wrapper to handle 3D input -> 1D Output (Target: Magnitude) for SHAP
    def predict_wrapper(data_flat):
        data_3d = data_flat.reshape(data_flat.shape[0], w_size, 5)
        pred = model.predict(data_3d, verbose=0)
        # Select the Magnitude feature from the output vector
        if pred.ndim == 3: return pred[:, 0, TARGET_FEAT_IDX]
        return pred[:, TARGET_FEAT_IDX] if pred.ndim == 2 else pred

    # Summarize background data using K-Means to speed up KernelSHAP
    X_tr_flat = X_tr.reshape(X_tr.shape[0], -1)
    background = shap.kmeans(X_tr_flat, 10)
    
    # Select random test samples
    indices = np.random.choice(X_te.shape[0], 10, replace=False)
    X_te_sample = X_te[indices]
    X_te_flat = X_te_sample.reshape(X_te_sample.shape[0], -1)
    
    # Run Explainer
    explainer = shap.KernelExplainer(predict_wrapper, background)
    shap_values = explainer.shap_values(X_te_flat, nsamples=500, silent=True)
    shap_3d = shap_values.reshape(-1, w_size, 5)
    
    # Plot A: Temporal Attention (Time Step Importance)
    temporal_imp = np.mean(np.abs(shap_3d), axis=(0, 2))
    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(w_size), temporal_imp, '-o', color='#e74c3c', markersize=4)
    plt.fill_between(np.arange(w_size), temporal_imp, color='#e74c3c', alpha=0.1)
    plt.title(f'Temporal Attention: {name} (w={w_size})')
    plt.xlabel('Time Step')
    plt.tight_layout()
    plt.savefig(f"SHAP_Temporal_{name}_w{w_size}.png", dpi=300)
    plt.close()
    
    # Plot B: Global Feature Importance
    global_imp = np.mean(np.abs(shap_3d), axis=(0, 1))
    idx = np.argsort(global_imp)
    plt.figure(figsize=(8, 4))
    plt.barh(range(5), global_imp[idx], color='#3498db')
    plt.yticks(range(5), [FEATURE_NAMES[i] for i in idx])
    plt.title(f'Global Importance: {name} (w={w_size})')
    plt.tight_layout()
    plt.savefig(f"SHAP_Global_{name}_w{w_size}.png", dpi=300)
    plt.close()
    print(f"    Plots Generated.")

# ==========================================
# 5. MAIN EXECUTION
# ==========================================
for w in [20, 50]:
    print(f"\n{'='*40}\n WINDOW SIZE {w}\n{'='*40}")
    
    X_tr, Y_tr = create_sequences(x_train_s, y_train_s, w, HORIZON)
    X_va, Y_va = create_sequences(x_val_s, y_val_s, w, HORIZON)
    X_te, Y_te = create_sequences(x_test_s, y_test_s, w, HORIZON)
    
    models = ["LSTM", "GRU", "TCN", "Transformer"]
    if w == 50:
        models += ["LSTM_Heavy", "GRU_Heavy"]
    
    for m in models:
        analyze_model(m, w, X_tr, Y_tr, X_va, Y_va, X_te, Y_te)

print("\n ALL TASKS COMPLETED SUCCESSFULLY.")