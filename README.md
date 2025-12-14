# Hierarchical vs. Sequential Processing for Earthquake Forecasting

**A Deep Learning Benchmark in the Zagros-Makran Transition Zone**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)

##  Overview
This repository contains the official implementation of the paper:
> **"Hierarchical vs. Sequential Processing: A Rigorous Assessment of Deep Learning Inductive Biases for Earthquake Forecasting"**

We conduct a rigorous benchmark of three architectural paradigms for spatiotemporal seismic forecasting:
1.  **Temporal Convolutional Networks (TCN):** Hierarchical processing (Proposed approach).
2.  **Recurrent Neural Networks (LSTM/GRU):** Sequential processing (Baseline).
3.  **Transformers:** Global attention mechanism (Baseline).

Our empirical results demonstrate that the **TCN architecture** possesses a hierarchical inductive bias better suited for seismic data, offering superior robustness to noise and generalization in medium-term forecasting ($w=50$) compared to RNNs and Transformers.

##  Project Structure
```text
├── data/
│   └── Final_Cleaned_Catalog_v2.csv   # (Required) Seismicity catalog input
├── models/                            # Directory for saving trained .keras models
├── train_benchmarks.py                # Step 1: Trains all models (LSTM, GRU, TCN, Transformer)
├── explain_shap.py                    # Step 2: Calculates SHAP values for all models (Global & Temporal)
├── plot_loss_comparison.py            # Step 3: Generates Figure 4 (Validation Loss Comparison)
├── plot_shap_temporal.py              # Step 3: Generates Figure 5 (Combined Temporal Attention)
├── plot_fmd.py                        # Analysis: Frequency-Magnitude Distribution (FMD) plotting
├── quick_test.py                      # Diagnostics: Verifies installation dependencies
├── requirements.txt                   # List of Python dependencies
└── README.md                          # Project documentation


##  Installation & Setup

1. Clone the Repository
git clone [https://github.com/YourUsername/Seismic-DL-Benchmark.git](https://github.com/YourUsername/Seismic-DL-Benchmark.git)
cd Seismic-DL-Benchmark
pip install -r requirements.txt


2. Install Dependencies
It is highly recommended to use a virtual environment (Python 3.8+).
```pip install -r requirements.txt```

3. Verify Environment
Run the quick diagnostic script to ensure TensorFlow, SHAP, and other libraries are correctly installed.
```python quick_test.py```

Usage Pipeline
To reproduce the study's results, please follow these steps in order:
Step 1: Training the Models
Trains all architectures from scratch across short-term (w=20) and medium-term (w=50) horizons.
```python train_benchmarks.py```
Output: Saves trained models to the root directory and prints MAE metrics.

Step 2: Feature Importance Analysis
Runs the SHAP explainer on the trained models to compute global and temporal feature importance. This step is essential for interpretability.
```python explain_shap.py```
Output: Generates individual SHAP plots for each model (useful for Supplementary Materials).

Step 3: Generating Paper Figures
Produces the final combined figures used in the manuscript:

Figure 4 (Validation Loss Comparison): Visualizes the convergence stability of the TCN compared to Transformers.
```python plot_loss_comparison.py```

Figure 5 (Causal Attention Profile): Combines SHAP results to demonstrate the TCN's focus on immediate precursors (Omori's Law).
```python plot_shap_temporal.py```

Key Findings & Visuals1.
 Convergence Stability (Figure 4)
 As shown below, the TCN (red) demonstrates stable convergence similar to RNNs, whereas the Transformer (grey) exhibits optimization volatility in longer sequences (w=50).
 (Note: Run plot_loss_comparison.py to generate this figure.)

 2. Physical Interpretability via SHAP (Figure 5)
 The SHAP analysis reveals that TCNs correctly prioritize immediate precursor events (consistent with Omori's Law), indicated by the high attention in the recent time steps (red highlight). In contrast, Transformers and LSTMs fail to establish a distinct temporal focus.
 (Note: Run plot_shap_temporal.py to generate this figure.)

🤝 Citation
If you use this code or dataset in your research, please cite:
@article{khalili2025hierarchical,
  title={Hierarchical vs. Sequential Processing: A Rigorous Assessment of Deep Learning Inductive Biases for Earthquake Forecasting},
  author={Khalili, Marzieh and Fotoohi, Ali},
  journal={Computers & Geosciences (Under Review)},
  year={2025}
}


📧 Contact
For questions or inquiries, please contact: Marzieh Khalili - marzieh-khalili@shirazu.ac.ir


