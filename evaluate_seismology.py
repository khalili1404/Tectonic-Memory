import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tcn import TCN
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, auc

np.random.seed(42)
tf.random.set_seed(42)

BASE_PATH = r'H:\YOURLOCATION'
DATA_FILE = os.path.join(BASE_PATH, 'journal of seismology', 'Final_Cleaned_Catalog_v2.csv')
MODEL_PATH = os.path.join(BASE_PATH, 'Model_TCN_w20.keras')

# 2. Data Loading and Preprocessing
def load_and_preprocess(data_path):
    """Loads the seismic catalog and applies log transformation to inter-event times."""
    df = pd.read_csv(data_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df[df['datetime'].dt.year >= 1998].reset_index(drop=True)
    df['time_diff_log'] = np.log1p(df['time_diff'])
    return df

df = load_and_preprocess(DATA_FILE)

feature_cols = ['LAT', 'LON', 'DEPTH', 'MAG', 'time_diff_log']
target_cols = ['time_diff', 'LAT', 'LON', 'MAG']

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

X_test, Y_test = [], []
for i in range(len(x_test_scaled) - WINDOW_SIZE - HORIZON + 1):
    X_test.append(x_test_scaled[i : i + WINDOW_SIZE])
    Y_test.append(y_test_scaled[i + WINDOW_SIZE : i + WINDOW_SIZE + HORIZON].flatten())

X_test = np.array(X_test)
Y_test = np.array(Y_test)

model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'TCN': TCN})
Y_pred = model.predict(X_test, verbose=0)

mag_idx = 3
y_test_physical = scaler_y.inverse_transform(Y_test.reshape(-1, HORIZON, 4)[:, 0, :])[:, mag_idx]
y_pred_physical = scaler_y.inverse_transform(Y_pred.reshape(-1, HORIZON, 4)[:, 0, :])[:, mag_idx]

TARGET_THRESHOLD = 3.5

y_true_binary = (y_test_physical >= TARGET_THRESHOLD).astype(int)
n_targets = np.sum(y_true_binary)

fpr, tpr, _ = roc_curve(y_true_binary, y_pred_physical)
roc_auc = auc(fpr, tpr)

sort_indices = np.argsort(y_pred_physical)[::-1]
sorted_actuals = y_true_binary[sort_indices]

tau = np.arange(1, len(y_true_binary) + 1) / len(y_true_binary)
nu = (n_targets - np.cumsum(sorted_actuals)) / n_targets

print("=========================================")
print(f" Target Events (M >= {TARGET_THRESHOLD}): {n_targets}")
print(f" TCN Model AUC: {roc_auc:.3f}")
print("=========================================")

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

axes[0].plot(tau, nu, color='#c0392b', lw=2.5, label='TCN Forecast Skill')
axes[0].plot([0, 1], [1, 0], 'k--', alpha=0.6, label='Random Guessing')
axes[0].fill_between(tau, nu, 1-tau, where=(nu < 1-tau), color='#2ecc71', alpha=0.15, label='Predictive Gain')

axes[0].set_xlim([0, 1])
axes[0].set_ylim([0, 1])
axes[0].set_xlabel(r'Alarm Rate ($\tau$)', fontsize=12, fontweight='bold')
axes[0].set_ylabel(r'Miss Rate ($\nu$)', fontsize=12, fontweight='bold')
axes[0].set_title(f'(a) Molchan Error Diagram ($M \\geq {TARGET_THRESHOLD}$)', fontsize=13)
axes[0].legend(loc='upper right', framealpha=0.9)
axes[0].grid(True, linestyle=':', alpha=0.7)

axes[1].plot(fpr, tpr, color='#2980b9', lw=2.5, label=f'TCN Model (AUC = {roc_auc:.3f})')
axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Random Guessing')

axes[1].set_xlim([0, 1])
axes[1].set_ylim([0, 1.05])
axes[1].set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
axes[1].set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
axes[1].set_title(f'(b) ROC Analysis ($M \\geq {TARGET_THRESHOLD}$)', fontsize=13)
axes[1].legend(loc='lower right', framealpha=0.9)
axes[1].grid(True, linestyle=':', alpha=0.7)

plt.tight_layout()

save_path_png = os.path.join(BASE_PATH, 'Figure6_Evaluation_M3.5_HighRes.png')
save_path_tiff = os.path.join(BASE_PATH, 'Figure6_Evaluation_M3.5_Print.tiff')

plt.savefig(save_path_png, dpi=600, bbox_inches='tight', format='png')
plt.savefig(save_path_tiff, dpi=600, bbox_inches='tight', format='tiff')

print("\n" + "="*65)
print(f" Successfully saved high-resolution PNG: \n {save_path_png}")
print(f" Successfully saved publication-ready TIFF: \n {save_path_tiff}")
print("="*65 + "\n")

plt.show()