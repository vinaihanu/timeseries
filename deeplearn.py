import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from statsmodels.tsa.arima.model import ARIMA
import math
import warnings
warnings.filterwarnings("ignore")

# ======================
# 1. Data Loading
# ======================
def load_financial_data(ticker="AAPL", start="2020-01-01", end="2023-01-01", features=None):
    df = yf.download(ticker, start=start, end=end)
    if features is None:
        features = ["Open", "High", "Low", "Close", "Volume"]
    data = df[features].fillna(method='ffill').values
    return data

# ======================
# 2. Dataset Class
# ======================
class TimeSeriesDataset(Dataset):
    """
    Dataset for sequence-to-sequence forecasting.
    Inputs:
        data: np.array, shape (n_samples, n_features)
        seq_len: input sequence length
        pred_len: prediction length
    """
    def __init__(self, data, seq_len=30, pred_len=1):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.X, self.y = self.create_sequences(data, seq_len, pred_len)

    def create_sequences(self, data, seq_len, pred_len):
        X, y = [], []
        for i in range(len(data) - seq_len - pred_len + 1):
            X.append(data[i:i+seq_len])
            y.append(data[i+seq_len:i+seq_len+pred_len])
        return torch.FloatTensor(X), torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ======================
# 3. Attention Mechanism
# ======================
class SelfAttention(nn.Module):
    """
    Self-Attention layer for sequence-to-sequence forecasting.
    """
    def __init__(self, input_dim, attention_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(input_dim, attention_dim)
        self.key = nn.Linear(input_dim, attention_dim)
        self.value = nn.Linear(input_dim, attention_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        scores = torch.bmm(Q, K.transpose(1,2)) / math.sqrt(Q.size(-1))  # (batch, seq_len, seq_len)
        weights = self.softmax(scores)
        out = torch.bmm(weights, V)  # (batch, seq_len, attention_dim)
        return out, weights

# ======================
# 4. Seq2Seq Model with Attention
# ======================
class Seq2SeqAttention(nn.Module):
    """
    Encoder-Decoder LSTM with Attention.
    """
    def __init__(self, input_dim, hidden_dim=64, attention_dim=32, num_layers=1, pred_len=1):
        super(Seq2SeqAttention, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.attention = SelfAttention(hidden_dim, attention_dim)
        self.decoder = nn.LSTM(attention_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)
        self.pred_len = pred_len

    def forward(self, x):
        # Encoder
        enc_out, (h, c) = self.encoder(x)
        # Attention
        attn_out, attn_weights = self.attention(enc_out)
        # Decoder
        dec_out, _ = self.decoder(attn_out)
        # Prediction: take last pred_len steps
        out = self.fc(dec_out[:, -self.pred_len:, :])
        return out, attn_weights

# ======================
# 5. Training Function
# ======================
def train_model(model, train_loader, epochs=50, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred, _ = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")
    return model

# ======================
# 6. Evaluation
# ======================
def evaluate_model(model, test_loader):
    model.eval()
    preds, trues = [], []
    attention_weights = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            pred, attn = model(X_batch)
            preds.append(pred.numpy())
            trues.append(y_batch.numpy())
            attention_weights.append(attn.numpy())
    preds = np.vstack(preds)
    trues = np.vstack(trues)
    
    # Reshape preds and trues to 2D arrays for sklearn metrics
    num_samples = preds.shape[0]
    preds = preds.reshape(num_samples, -1)
    trues = trues.reshape(num_samples, -1)

    rmse = np.sqrt(mean_squared_error(trues, preds))
    mape = mean_absolute_percentage_error(trues, preds)
    return preds, trues, attention_weights, rmse, mape

# ======================
# 7. Baseline: ARIMA
# ======================
def arima_baseline(data, pred_len=1):
    # Univariate baseline: last feature
    series = data[:, -1]  # e.g., Close price
    train_size = int(len(series) * 0.8)
    train, test = series[:train_size], series[train_size:]
    history = list(train)
    predictions = []
    for t in range(len(test)):
        model = ARIMA(history, order=(5,1,0))
        model_fit = model.fit()
        yhat = model_fit.forecast(steps=pred_len)[0]
        predictions.append(yhat)
        history.append(test[t])
    predictions = np.array(predictions).reshape(-1,1)
    rmse = np.sqrt(mean_squared_error(test.reshape(-1,1), predictions))
    mape = mean_absolute_percentage_error(test.reshape(-1,1), predictions)
    return predictions, rmse, mape

# ======================
# 8. Main Script
# ======================
if __name__ == "__main__":
    # Load data
    data = load_financial_data(ticker="AAPL")
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Train-test split
    split = int(len(data_scaled)*0.8)
    train_data = data_scaled[:split]
    test_data = data_scaled[split:]

    seq_len = 30
    pred_len = 1
    batch_size = 32

    train_dataset = TimeSeriesDataset(train_data, seq_len, pred_len)
    test_dataset = TimeSeriesDataset(test_data, seq_len, pred_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize and train Seq2SeqAttention
    input_dim = data.shape[1]
    model = Seq2SeqAttention(input_dim=input_dim)
    model = train_model(model, train_loader, epochs=20, lr=0.001)

    # Evaluate
    preds, trues, attn_weights, rmse, mape = evaluate_model(model, test_loader)
    print(f"Attention Model RMSE: {rmse:.4f}, MAPE: {mape:.4f}")

    # Plot attention weights for first batch
    plt.figure(figsize=(8,6))
    plt.imshow(attn_weights[0][0], cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title("Attention Weights (First Sample)")
    plt.xlabel("Historical Time Steps")
    plt.ylabel("Sequence Step")
    plt.show()

    # Baseline: ARIMA
    arima_pred, arima_rmse, arima_mape = arima_baseline(data_scaled)
    print(f"ARIMA RMSE: {arima_rmse:.4f}, MAPE: {arima_mape:.4f}")
     





