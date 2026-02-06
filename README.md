# Advanced Time Series Forecasting with Seq2Seq Attention

This project implements an **advanced multivariate time series forecasting pipeline** using deep learning with a **Sequence-to-Sequence (Seq2Seq) LSTM architecture augmented by a custom self-attention mechanism**. The implementation also includes strong baseline models (Standard LSTM and ARIMA) and emphasizes **model interpretability through attention weight visualization**.

The code is written as a **single, production-quality Python file**, suitable for advanced coursework or research-oriented experimentation.

---

## 📌 Project Objectives

* Build a multivariate time series forecasting system with **5+ features and 1000+ observations**
* Implement a **custom self-attention mechanism** on top of an encoder–decoder LSTM
* Compare performance against simpler baselines (Standard LSTM, ARIMA)
* Go beyond numerical metrics by **interpreting learned attention weights**
* Deliver clean, modular, and well-documented code

---

## 📁 Project Structure

```
.
├── attention_time_series_forecasting.py   # Main Python implementation
├── README.md                              # Project documentation
```

All functionality (data generation, modeling, training, evaluation, visualization) is contained in **one Python file** as required.

---

## 📊 Dataset Description

The dataset is **synthetically generated** to simulate complex real-world behavior:

* **1500 time steps**
* **5 continuous features**
* Nonlinear dynamics using sine and cosine waves
* Additive Gaussian noise for realism

Each feature is generated as:

* A combination of sinusoids with different frequencies
* Independent noise injection

This design ensures:

* Temporal dependencies
* Cross-feature variability
* Suitability for attention-based modeling

---

## 🧠 Model Architectures

### 1. Seq2Seq LSTM with Self-Attention (Primary Model)

**Architecture:**

* Encoder: LSTM processes historical input window
* Self-Attention: Learns importance of historical time steps
* Decoder: LSTM generates multi-step forecasts
* Fully Connected Layer: Maps hidden states to output features

**Key Advantage:**

The attention mechanism provides **interpretability**, allowing inspection of which past time steps influence future predictions.

---

### 2. Standard LSTM (Baseline)

* Single LSTM encoder
* No attention mechanism
* Uses final hidden state for forecasting

Used to quantify performance gains from attention.

---

### 3. ARIMA (Baseline)

* Classical statistical model
* Applied to a **single feature only**
* Serves as a traditional forecasting benchmark

---

## ⚙️ Training Configuration

| Parameter        | Value |
| ---------------- | ----- |
| Input Length     | 30    |
| Output Horizon   | 10    |
| Hidden Dimension | 64    |
| Batch Size       | 32    |
| Epochs           | 15    |
| Optimizer        | Adam  |
| Loss Function    | MSE   |

---

## 📈 Evaluation Metrics

The following metrics are computed on the test set:

* **RMSE (Root Mean Squared Error)**
* **MAPE (Mean Absolute Percentage Error)**

All metrics are computed over **all features and forecast horizons** for deep learning models.

---

## 🔍 Attention Weight Interpretation

The self-attention module produces a **(sequence × sequence)** attention matrix.

Visualization highlights:

* Rows: Query time steps
* Columns: Historical encoder time steps
* Color intensity: Attention strength

This allows qualitative analysis such as:

* Identifying critical historical windows
* Observing periodic dependencies
* Validating model focus during regime shifts

---

## ▶️ How to Run

### 1. Install Dependencies

```bash
pip install numpy pandas matplotlib torch scikit-learn statsmodels
```

### 2. Execute the Script

```bash
python attention_time_series_forecasting.py
```

The script will:

1. Generate the dataset
2. Train all models
3. Print evaluation metrics
4. Display attention heatmaps

---

## ✅ Expected Outputs

* Training loss per epoch
* RMSE and MAPE for:

  * Attention-based Seq2Seq model
  * Standard LSTM baseline
* ARIMA RMSE for one feature
* Attention heatmap visualization

---

## 🚀 Possible Extensions

* Multi-head attention
* Transformer-based forecasting
* Real-world datasets (e.g., stock prices via `yfinance`)
* Probabilistic forecasting (quantile loss)
* Hyperparameter optimization

---
## ✍️ Author

**Deepa**
Advanced Time Series Forecasting Project
