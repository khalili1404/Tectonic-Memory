# Tectonic Memory and Short-Term Earthquake Forecasting in the Zagros-Makran Belt: A Seismological Validation of Deep Sequence Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)

## Overview
This repository contains the official implementation, datasets, and evaluation frameworks for the paper:
> **"Tectonic Memory and Short-Term Earthquake Forecasting in the Zagros-Makran Belt: A Seismological Validation of Deep Sequence Models"** (Submitted to the *Journal of Seismology*).

Moving beyond standard machine-learning regression metrics, this study conducts a rigorous benchmark of deep sequence models (TCN, LSTM, GRU, Transformer) heavily grounded in **applied seismology**. We specifically evaluate these parameter-controlled architectures (~57k parameters) in the highly chaotic tectonic regime of the Zagros-Makran transition zone. 

**Key Contributions:**
1. **Operational Seismological Validation:** Utilizing the **Molchan Error Diagram** and **ROC Analysis** to quantify true physical predictive gain across multiple magnitude thresholds ($M \ge 3.5, 4.0, 4.5$).
2. **ETAS Benchmarking:** Directly comparing TCN magnitude predictability against the classical Epidemic-Type Aftershock Sequence (ETAS) theoretical limit (AUC = 0.50).
3. **Physical Interpretability:** Using SHAP analysis to demonstrate how hierarchical causal convolutions inherently capture **Omori's Law** decay.
4. **The Data Starvation Paradigm:** Identifying the fundamental bottleneck of AI-driven seismology when predicting major ruptures at the historical completeness magnitude.

---

## Project Structure
```text
├── data/
│   └── Final_Cleaned_Catalog_v2.csv   # (Required) Homogeneous seismicity catalog (post-1998)
├── models/                            # Directory for saving trained .keras models
├── train_benchmarks.py                # Step 1: Trains all controlled models (LSTM, GRU, TCN, Transformer)
├── benchmark_etas_tcn.py              # Step 2: Benchmarks TCN against the classical ETAS model
├── explain_shap.py                    # Step 3: Calculates SHAP values for interpretability
├── plot_loss_comparison.py            # Step 4: Generates Figure 4 (Validation Loss Comparison & Stability)
├── evaluate_seismology.py             # Step 5: Generates Figure 5 (Molchan & ROC Analysis for target thresholds)
├── plot_shap_temporal.py              # Step 6: Generates Figure 6 & 7 (Temporal Attention & Omori's Law)
├── requirements.txt                   # List of Python dependencies
└── README.md                          # Project documentation
```

##  Installation & Setup

1. Clone the Repository

```bash
git clone https://anonymous.4open.science/r/Hierarchical-Vs-Sequential-Processing-8441
cd Seismic-DL-Benchmark
pip install -r requirements.txt
```

2. Install Dependencies

It is highly recommended to use a virtual environment (Python 3.8+).
```bash
pip install -r requirements.txt
```

3. Verify Environment

Run the quick diagnostic script to ensure TensorFlow, SHAP, and other libraries are correctly installed.

```bash
python quick_test.py
```

## Usage Pipeline

To reproduce the study's results, please follow these steps in order:

1. Model Training

Trains all architectures under strict chronological splitting to prevent data leakage.

```bash
python train_benchmarks.py
```
Output: Saves trained models to the root directory and prints MAE metrics.

2. ETAS vs. TCN Benchmarking

Evaluates the TCN architecture against the theoretical limits of the ETAS model to prove magnitude predictability.

```bash
python benchmark_etas_tcn.py
```

Output: Generates ROC-AUC comparisons demonstrating the TCN surpassing the ETAS 0.50 baseline.

3. Seismological Validation (Molchan & ROC)

Evaluates the optimal TCN model ($w=20$) against random guessing for operational forecasting skill.

```bash
python evaluate_seismology.py
```

Output: Generates Figure 5 (AUC = 0.560 for $M \ge 3.5$)

4. Physical Interpretability (SHAP)

Runs the SHAP explainer to compute temporal feature importance, proving the model's alignment with tectonic physics.

```bash
python explain_shap.py
python plot_shap_temporal.py
```

Output: Generates Figure 7 (Causal Attention Profile highlighting recent clustering dynamics).

5. Convergence & Stability Analysis

Visualizes the validation loss to demonstrate the TCN's robustness against noise accumulation over extended sequences ($w=50$).

```bash
python plot_loss_comparison.py
```
Output: Generates Figure 4 and Supplementary Figures S1 & S2.

## Key Findings

1. `ETAS Limit Broken`:TCN successfully breaks the magnitude-independence barrier of classical statistical models (ETAS), improving magnitude predictability from a random baseline of AUC=0.50 to a meaningful 0.560.

2. `Seismological Predictive Gain`: The TCN strictly outperforms random guessing in the Molchan error diagram, optimizing hit rates during periods of minimal spatial-temporal alarm coverage for moderate seismicity.

3. `Alignment with Earthquake Physics`: SHAP analysis reveals that the TCN correctly prioritizes immediate precursor events (consistent with Omori's Law), whereas Transformers and LSTMs fail to establish a distinct temporal focus.

4. `The Data Starvation Bottleneck`: While the model mathematically recovers diagnostic skill at higher thresholds ($M_c \ge 4.5$), operational forecasting of large-scale ruptures is fundamentally constrained by the inherent sparsity of the instrumental catalog, rather than architectural limits.

## Citation

### If you use this code or dataset in your research, please cite:

```bibtex
@article{khalili2025tectonic,
  title={Tectonic Memory and Short-Term Earthquake Forecasting in the Zagros-Makran Belt: A Seismological Validation of Deep Sequence Models},
  author={Khalili, Marzieh and Fotoohi, Ali},
  journal={Journal of Seismology},
  year={2026}
}
```
