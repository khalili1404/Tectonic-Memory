"""
Quick System Check for Earthquake Forecasting Models
Description:
    This script generates dummy data and runs a single training epoch
    to verify that TensorFlow, SHAP, and all dependencies are correctly installed.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def run_quick_check():
    print(">>> 1. Checking Dependencies...")
    try:
        import shap
        import matplotlib.pyplot as plt
        from sklearn.preprocessing import MinMaxScaler
        print("    [+] Libraries imported successfully.")
    except ImportError as e:
        print(f"    [-] Error: Missing library. {e}")
        return

    print(">>> 2. Generating Dummy Seismic Data...")
    # Simulate a small catalog (N=100, Features=5)
    data = np.random.rand(100, 5)
    targets = np.random.rand(100, 1)
    
    # Reshape for Time Series (Batch, Window=10, Features=5)
    X = data.reshape(10, 10, 5)
    Y = targets[:10]
    print(f"    [+] Data shape created: {X.shape}")

    print(">>> 3. Building & Training Test Model...")
    try:
        model = Sequential([
            Input(shape=(10, 5)),
            LSTM(8, return_sequences=False),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, Y, epochs=1, verbose=0)
        print("    [+] Model training test passed.")
    except Exception as e:
        print(f"    [-] Model Error: {e}")
        return

    print("\n SYSTEM CHECK PASSED. You are ready to run the full benchmarks.")

if __name__ == "__main__":
    run_quick_check()