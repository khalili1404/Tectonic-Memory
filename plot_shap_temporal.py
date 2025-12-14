import os
import warnings
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# ==========================================
# 1. SETUP & STYLE
# ==========================================
def set_seeds(seed=42):
    """Sets random seeds for reproducibility."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seeds()
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12
plt.rcParams['figure.dpi'] = 300

DATA_FILE = 'Final_Cleaned_Catalog_v2.csv'
CUTOFF_YEAR = 1998
HORIZON = 10
TARGET_FEAT_IDX = 3  # Magnitude Index

# ==========================================
# 2. DATA PREPARATION
# ==========================================
def get_data():
    """Load and preprocess data efficiently."""
    try:
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        print(f"Error: {DATA_FILE} not found.")
        return None, None, None

    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df[df['datetime'].dt.year >= CUTOFF_YEAR].reset_index(drop=True)
    df['time_diff_log'] = np.log1p(df['time_diff'])
    
    cols_x = ['LAT', 'LON', 'DEPTH', 'MAG', 'time_diff_log']
    cols_y = ['time_diff', 'LAT', 'LON', 'MAG']
    
    # Chronological Split (80-10-10)
    n = len(df)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)
    train_df = df.iloc[:train_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    # Fit Scaler ONLY on Training Data
    scaler_x = MinMaxScaler().fit(train_df[cols_x])
    scaler_y = MinMaxScaler().fit(train_df[cols_y])
    
    x_train = scaler_x.transform(train_df[cols_x])
    x_test = scaler_x.transform(test_df[cols_x])
    y_test = scaler_y.transform(test_df[cols_y])
    
    return x_train, x_test, y_test

def create_sequences(features, targets, w, h):
    """Generate sequences compatible with the models."""
    X, Y = [], []
    for i in range(len(features) - w - h + 1):
        X.append(features[i : i + w])
        Y.append(targets[i + w : i + w + h].flatten())
    return np.array(X), np.array(Y)

# ==========================================
# 3. PLOTTING ENGINE (SHAP ONLY)
# ==========================================
def run_shap_for_window(window_size, x_train, x_test, y_test):
    print(f"\n{'='*40}")
    print(f"Processing SHAP for Window Size: {window_size}")
    print(f"{'='*40}")

    # 1. Create Sequences
    X_test_seq, _ = create_sequences(x_test, y_test, window_size, HORIZON)
    
    # Select distinct test samples (Fixed Seed for Reproducibility)
    np.random.seed(123) 
    if len(X_test_seq) < 10:
        print("Not enough test data for SHAP.")
        return
        
    indices = np.random.choice(X_test_seq.shape[0], 10, replace=False)
    X_sample = X_test_seq[indices]
    X_flat = X_sample.reshape(X_sample.shape[0], -1)
    
    # 2. Create Background (Dummy Targets for Shape Matching)
    # Important: targets must match x_train length for create_sequences to work
    dummy_targets = np.zeros((len(x_train), 4)) 
    X_train_seq, _ = create_sequences(x_train, dummy_targets, window_size, HORIZON)
    
    # K-Means Summary for Speed
    X_train_flat = X_train_seq.reshape(X_train_seq.shape[0], -1)
    # Fixed seed for K-Means background selection
    np.random.seed(42)
    idx_bg = np.random.choice(X_train_flat.shape[0], 200, replace=False) 
    background = shap.kmeans(X_train_flat[idx_bg], 10)
    
    # 3. Define Models to Compare
    # Note: LSTM is standard for w=20, Heavy for w=50
    lstm_name = "LSTM_Heavy" if window_size == 50 else "LSTM"
    
    models_to_compare = [
        ('TCN', f'Model_TCN_w{window_size}.keras', '#d62728', '-o'),       
        ('Transformer', f'Model_Transformer_w{window_size}.keras', '#7f7f7f', '--s'), 
        (lstm_name, f'Model_{lstm_name}_w{window_size}.keras', '#1f77b4', ':^')   
    ]
    
    plt.figure(figsize=(10, 6))
    
    for name, path, color, style in models_to_compare:
        print(f"   Analyzing {name}...")
        
        # Handle case-sensitive filenames
        if not os.path.exists(path):
            if os.path.exists(path.replace("Model_", "model_")):
                path = path.replace("Model_", "model_")
            else:
                print(f"    File not found: {path}. Skipping.")
                continue
                
        try:
            model = load_model(path)
            
            # Wrapper function for SHAP
            def predict_wrapper(data_flat):
                data_3d = data_flat.reshape(data_flat.shape[0], window_size, 5)
                pred = model.predict(data_3d, verbose=0)
                if pred.ndim == 3: return pred[:, 0, TARGET_FEAT_IDX]
                return pred[:, TARGET_FEAT_IDX]

            explainer = shap.KernelExplainer(predict_wrapper, background)
            # Reduced nsamples slightly for speed/stability
            shap_vals = explainer.shap_values(X_flat, nsamples=200, silent=True)
            shap_3d = shap_vals.reshape(-1, window_size, 5)
            
            # Compute Temporal Attention
            temporal_imp = np.mean(np.abs(shap_3d), axis=(0, 2))
            # Min-Max Normalization for comparison
            if temporal_imp.max() > temporal_imp.min():
                temporal_imp = (temporal_imp - temporal_imp.min()) / (temporal_imp.max() - temporal_imp.min())
            
            plt.plot(range(window_size), temporal_imp, style, label=name, color=color, linewidth=2, markersize=5, alpha=0.8)
            
        except Exception as e:
            print(f"   Error in {name}: {e}")

    plt.title(f'Normalized Temporal Attention Profile (w={window_size})', fontweight='bold')
    plt.xlabel(f'Time Step (0=Oldest, {window_size-1}=Recent)', fontweight='bold')
    plt.ylabel('Normalized Feature Importance', fontweight='bold')
    plt.legend(frameon=True, shadow=True)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Add Highlight only for w=50 (Interpretation Zone)
    if window_size == 50:
        plt.axvspan(40, 49, color='#d62728', alpha=0.1)
        plt.text(41, 0.1, 'Immediate\nPrecursors', color='#d62728', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    filename = f'Figure_SHAP_w{window_size}.png'
    plt.savefig(filename, dpi=300)
    print(f" Saved: {filename}")

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # 1. Load Data
    x_train, x_test, y_test = get_data()
    
    if x_train is not None:
        # 2. Run SHAP for BOTH windows (Fixed logic)
        for w in [20, 50]:
            run_shap_for_window(w, x_train, x_test, y_test)
    else:
        print("Skipping execution due to missing data.")