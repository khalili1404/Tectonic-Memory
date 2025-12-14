import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (LSTM, Dense, Dropout, MultiHeadAttention, 
                                     LayerNormalization, Input, GlobalAveragePooling1D, 
                                     Add, Embedding, Conv1D)
from tensorflow.keras.callbacks import EarlyStopping

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 300
plt.rcParams['lines.linewidth'] = 2

DATA_FILE = 'Final_Cleaned_Catalog_v2.csv'
CUTOFF_YEAR = 1998
HORIZON = 10
EPOCHS = 40
BATCH_SIZE = 16

warnings.filterwarnings('ignore')

# ==========================================
# 2. DATA PREPARATION
# ==========================================
def create_sequences(features, targets, w, h):
    X, Y = [], []
    for i in range(len(features) - w - h + 1):
        X.append(features[i : i + w])
        Y.append(targets[i + w : i + w + h].flatten())
    return np.array(X), np.array(Y)

def get_data_splits(df):
    n = len(df)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)
    return df.iloc[:train_end].copy(), df.iloc[train_end:val_end].copy(), df.iloc[val_end:].copy()

print("--- Loading Data ---")
try:
    df = pd.read_csv(DATA_FILE)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df[df['datetime'].dt.year >= CUTOFF_YEAR].reset_index(drop=True)
except:
    print("WARNING: Dataset not found. Generating DUMMY data for demonstration.")
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

train_df, val_df, test_df = get_data_splits(df)
scaler_x = MinMaxScaler().fit(train_df[cols_x])
scaler_y = MinMaxScaler().fit(train_df[cols_y])

x_train_s, y_train_s = scaler_x.transform(train_df[cols_x]), scaler_y.transform(train_df[cols_y])
x_val_s, y_val_s = scaler_x.transform(val_df[cols_x]), scaler_y.transform(val_df[cols_y])

# ==========================================
# 3. MODEL ARCHITECTURES
# ==========================================
def build_model(name, w, n_feat, out_dim):
    if name == "TCN":
        inputs = Input(shape=(w, n_feat))
        x = inputs
        for d in [1, 2, 4, 8, 16]:
            conv = Conv1D(64, 3, dilation_rate=d, padding='causal', activation='relu')(x)
            if x.shape[-1] != 64: 
                x = Conv1D(64, 1)(x)
            x = Add()([x, conv])
            x = Dropout(0.2)(x)
        x = GlobalAveragePooling1D()(x)
        x = Dense(64, activation='relu')(x)
        return Model(inputs, Dense(out_dim)(x))
    
    elif name == "Transformer":
        inputs = Input(shape=(w, n_feat))
        d_model = 64
        x = Dense(d_model)(inputs)
        pos = tf.range(start=0, limit=w, delta=1)
        emb = Embedding(w, d_model, trainable=False)(pos)
        x = Add()([x, tf.expand_dims(emb, axis=0)])
        att = MultiHeadAttention(4, 16)(x, x)
        x = LayerNormalization(epsilon=1e-6)(Add()([x, att]))
        ffn = Sequential([Dense(d_model, "relu"), Dense(d_model)])(x)
        x = LayerNormalization(epsilon=1e-6)(Add()([x, ffn]))
        x = GlobalAveragePooling1D()(x)
        x = Dropout(0.2)(x)
        x = Dense(64, "relu")(x)
        return Model(inputs, Dense(out_dim)(x))

    elif name == "LSTM_Heavy":
        return Sequential([
            Input(shape=(w, n_feat)),
            LSTM(64, return_sequences=True), Dropout(0.2),
            LSTM(64, return_sequences=False), Dropout(0.2),
            Dense(64, activation='relu'), Dense(out_dim)
        ])
        
    elif name == "LSTM":
        return Sequential([
            Input(shape=(w, n_feat)),
            LSTM(64, return_sequences=False), Dropout(0.2),
            Dense(64, activation='relu'), Dense(out_dim)
        ])

# ==========================================
# 4. TRAINING LOOP
# ==========================================
history_storage = {}

for w in [20, 50]:
    print(f"\n>>> Running for Window Size: {w}")
    X_tr, Y_tr = create_sequences(x_train_s, y_train_s, w, HORIZON)
    X_va, Y_va = create_sequences(x_val_s, y_val_s, w, HORIZON)
    
    if w == 20:
        models_to_run = ["TCN", "Transformer", "LSTM"]
    else:
        models_to_run = ["TCN", "Transformer", "LSTM_Heavy"]
        
    for m_name in models_to_run:
        print(f"   Training {m_name}...")
        tf.keras.backend.clear_session()
        
        model = build_model(m_name, w, 5, 40)
        model.compile(optimizer='adam', loss='mse')
        
        hist = model.fit(X_tr, Y_tr, validation_data=(X_va, Y_va),
                         epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0,
                         callbacks=[EarlyStopping(patience=10, restore_best_weights=True)])
        
        key = f"{m_name}_w{w}"
        history_storage[key] = hist.history['val_loss']
        
        # Save history for reproducibility
        pd.DataFrame(hist.history).to_csv(f"History_{key}.csv")

# ==========================================
# 5. GENERATE PLOTS
# ==========================================
print("\n>>> Generating Final Publication Figure...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

styles = {
    'TCN': {'color': '#d62728', 'ls': '-', 'marker': 'o', 'lw': 2.5, 'label': 'TCN (Proposed)'},
    'Transformer': {'color': '#7f7f7f', 'ls': '--', 'marker': '', 'lw': 2, 'label': 'Transformer'},
    'LSTM': {'color': '#1f77b4', 'ls': ':', 'marker': '', 'lw': 2, 'label': 'LSTM'},
    'LSTM_Heavy': {'color': '#1f77b4', 'ls': ':', 'marker': '', 'lw': 2, 'label': 'LSTM (Heavy)'}
}

# --- Plot (a) w=20 ---
ax = ax1
# Plot TCN last to ensure visibility on top
for m_name in ["Transformer", "LSTM", "TCN"]:
    key = f"{m_name}_w20"
    if key in history_storage:
        data = history_storage[key]
        epochs = range(1, len(data) + 1)
        s = styles.get(m_name, {})
        ax.plot(epochs, data, color=s['color'], linestyle=s['ls'], 
                linewidth=s['lw'], marker=s['marker'], markevery=2, label=s['label'])

ax.set_title('(a) Short-Term Forecasting (w=20)', fontweight='bold')
ax.set_xlabel('Epoch')
ax.set_ylabel('Validation Loss (MSE)')
ax.legend()
ax.grid(True, linestyle=':', alpha=0.6)

# --- Plot (b) w=50 ---
ax = ax2
for m_name in ["Transformer", "LSTM_Heavy", "TCN"]:
    key = f"{m_name}_w50"
    if key in history_storage:
        data = history_storage[key]
        epochs = range(1, len(data) + 1)
        s = styles.get(m_name, {})
        ax.plot(epochs, data, color=s['color'], linestyle=s['ls'], 
                linewidth=s['lw'], marker=s['marker'], markevery=2, label=s['label'])

ax.set_title('(b) Medium-Term Forecasting (w=50)', fontweight='bold')
ax.set_xlabel('Epoch')
ax.legend()
ax.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig('Figure_4_Combined_Loss_Real.jpg', dpi=300)
print(" Figure Saved: Figure_4_Combined_Loss_Real.jpg")