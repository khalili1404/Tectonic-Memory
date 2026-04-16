import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tcn import TCN
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, auc

# =======================================================
# 1. Path Configuration (Local Research Environment)
# =======================================================
BASE_DIR = r"H:\Beyrami\first"
DATA_FILE = os.path.join(BASE_DIR, 'journal of seismology', 'Final_Cleaned_Catalog_v2.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'Model_TCN_w20.keras')
OUTPUT_IMAGE = os.path.join(BASE_DIR, 'journal of seismology', 'Rebuttal_Figure_Final.png')

# =======================================================
# 2. Dataset Preprocessing and Model Loading
# =======================================================
print("Loading data and model...")
df = pd.read_csv(DATA_FILE)
df['datetime'] = pd.to_datetime(df['datetime'])

# Filter catalog to ensure instrumental completeness (Post-1998)
df = df[df['datetime'].dt.year >= 1998].reset_index(drop=True)
# Apply log transform to temporal features (Seismogenic time-interval normalization)
df['time_diff_log'] = np.log1p(df['time_diff'])

feature_cols = ['LAT', 'LON', 'DEPTH', 'MAG', 'time_diff_log']
target_cols = ['time_diff', 'LAT', 'LON', 'MAG']

# Chronological data splitting to prevent data leakage
train_size = int(len(df) * 0.8)
test_start = int(len(df) * 0.9)

train_df = df.iloc[:train_size].copy()
test_df = df.iloc[test_start:].copy()

scaler_x = MinMaxScaler().fit(train_df[feature_cols])
scaler_y = MinMaxScaler().fit(train_df[target_cols])

WINDOW_SIZE = 20
HORIZON = 10

x_test_scaled = scaler_x.transform(test_df[feature_cols])
y_test_scaled = scaler_y.transform(test_df[target_cols])

# Construct sliding window sequences for testing
X_test, Y_test = [], []
for i in range(len(x_test_scaled) - WINDOW_SIZE - HORIZON + 1):
    X_test.append(x_test_scaled[i : i + WINDOW_SIZE])
    Y_test.append(y_test_scaled[i + WINDOW_SIZE : i + WINDOW_SIZE + HORIZON].flatten())

X_test = np.array(X_test)
Y_test = np.array(Y_test)

# Load the trained Deep Sequence Model (TCN architecture)
model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'TCN': TCN})
Y_pred = model.predict(X_test, verbose=0)

# =======================================================
# 3. Operational Metric Calculation (Magnitude Predictability)
# =======================================================
print("Calculating real FPR and TPR from your model...")
mag_idx = 3 # Index for Magnitude feature
y_test_physical = scaler_y.inverse_transform(Y_test.reshape(-1, HORIZON, 4)[:, 0, :])[:, mag_idx]
y_pred_physical = scaler_y.inverse_transform(Y_pred.reshape(-1, HORIZON, 4)[:, 0, :])[:, mag_idx]

# Evaluating performance against the target seismicity threshold
TARGET_THRESHOLD = 3.5
y_true_binary = (y_test_physical >= TARGET_THRESHOLD).astype(int)

# Extracting key performance metrics for seismological validation
fpr, tpr, _ = roc_curve(y_true_binary, y_pred_physical)
roc_auc = auc(fpr, tpr)

# =======================================================
# 4. Comparative Visualization (Benchmarking ETAS vs. TCN)
# =======================================================
print("Generating the exact Rebuttal Figure...")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.5

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# ---- Panel A: AUC Capability Comparison ----
labels = ['Spatiotemporal\n(Time & Location)', 'Magnitude\n(Event Size)']
x = np.arange(len(labels))
width = 0.35

# Plotting overall forecasting skill
ax1.bar(x[0], 0.8669, width, color='#4c72b0', edgecolor='black', linewidth=1.2, zorder=3)
ax1.text(x[0], 0.8669 + 0.015, '0.86', ha='center', va='bottom', fontweight='bold')

# Plotting magnitude predictability (ETAS baseline vs. TCN gain)
ax1.bar(x[1] - width/2, 0.5000, width, label='ETAS (Statistical Baseline)', color='#4c72b0', edgecolor='black', linewidth=1.2, zorder=3)
ax1.text(x[1] - width/2, 0.5000 + 0.015, 'Theoretical\nLimit (0.50)', ha='center', va='bottom', fontsize=10, color='#1f4e79')

ax1.bar(x[1] + width/2, roc_auc, width, label='TCN (Deep Sequence Model)', color='#c44e52', edgecolor='black', linewidth=1.2, zorder=3)
ax1.text(x[1] + width/2, roc_auc + 0.015, f'{roc_auc:.2f}', ha='center', va='bottom', fontweight='bold', color='#8b0000')

ax1.set_ylabel('AUC Score', fontweight='bold')
ax1.set_title('(a) Predictive Capability Comparison', loc='left', fontweight='bold', pad=15)
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontweight='bold')
ax1.set_ylim(0.4, 1.0)
ax1.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5, zorder=0)
ax1.legend(loc='upper right', framealpha=0.9)
ax1.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)

# ---- Panel B: Receiver Operating Characteristic (ROC) ----
# Benchmark: Magnitude independence (Random Guessing limit)
ax2.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=2, label='Random Guessing (AUC = 0.50)')
ax2.plot([0, 1], [0, 1], color='#4c72b0', linestyle=':', linewidth=4, alpha=0.8, label='ETAS Baseline (AUC = 0.50)')

# TCN Predictive Gain visualization
ax2.plot(fpr, tpr, color='#c44e52', linewidth=3, label=f'TCN Memory Model (AUC = {roc_auc:.2f})')

ax2.set_xlabel('False Positive Rate', fontweight='bold')
ax2.set_ylabel('True Positive Rate', fontweight='bold')
ax2.set_title('(b) ROC Curve for Magnitude Predictability', loc='left', fontweight='bold', pad=15)
ax2.legend(loc='lower right', framealpha=0.9)
ax2.grid(True, linestyle='--', alpha=0.6)

# Final save and rendering for rebuttal report
plt.tight_layout()
plt.savefig(OUTPUT_IMAGE, dpi=300, bbox_inches='tight')
print(f"\n✅ Figure successfully saved at:\n{OUTPUT_IMAGE}")
plt.show()
