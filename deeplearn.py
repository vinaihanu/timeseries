import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from statsmodels.tsa.arima.model import ARIMA


# ===============================
# Data Generation & Preprocessing
# ===============================

def generate_multivariate_time_series(
    n_steps: int = 1500,
    n_features: int = 5,
    noise_std: float = 0.1
) -> pd.DataFrame:
    """
    Generates a synthetic multivariate time series with nonlinear interactions.

    Returns:
        pd.DataFrame: shape (n_steps, n_features)
    """
    t = np.arange(n_steps)

    data = []
    for i in range(n_features):
        signal = (
            np.sin(0.01 * t * (i + 1)) +
            np.cos(0.015 * t * (i + 2)) +
            np.random.normal(0, noise_std, size=n_steps)
        )
        data.append(signal)

    data = np.vstack(data).T
    columns = [f"feature_{i}" for i in range(n_features)]
    return pd.DataFrame(data, columns=columns)


class TimeSeriesDataset(Dataset):
    """
    Sliding-window dataset for sequence-to-sequence forecasting.
    """

    def __init__(self, data: np.ndarray, input_len: int, output_len: int):
        self.data = data
        self.input_len = input_len
        self.output_len = output_len

    def __len__(self):
        return len(self.data) - self.input_len - self.output_len

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.input_len]
        y = self.data[idx + self.input_len:idx + self.input_len + self.output_len]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# ===============================
# Attention Mechanism
# ===============================

class SelfAttention(nn.Module):
    """
    Scaled dot-product self-attention layer.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.scale = np.sqrt(hidden_dim)

    def forward(self, encoder_outputs):
        """
        Args:
            encoder_outputs: (batch, seq_len, hidden_dim)
        """
        Q = self.query(encoder_outputs)
        K = self.key(encoder_outputs)
        V = self.value(encoder_outputs)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_weights = torch.softmax(scores, dim=-1)

        context = torch.matmul(attention_weights, V)
        return context, attention_weights


# ===============================
# Seq2Seq Model with Attention
# ===============================

class Seq2SeqAttentionModel(nn.Module):
    """
    Encoder-Decoder LSTM with self-attention.
    """

    def __init__(self, n_features, hidden_dim, output_len):
        super().__init__()

        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.attention = SelfAttention(hidden_dim)

        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, n_features)
        self.output_len = output_len

    def forward(self, x):
        encoder_outputs, _ = self.encoder(x)
        context, attention_weights = self.attention(encoder_outputs)

        decoder_input = context[:, -1:, :].repeat(1, self.output_len, 1)
        decoder_outputs, _ = self.decoder(decoder_input)

        output = self.fc(decoder_outputs)
        return output, attention_weights


# ===============================
# Baseline LSTM Model
# ===============================

class StandardLSTM(nn.Module):
    """
    Simple LSTM baseline without attention.
    """

    def __init__(self, n_features, hidden_dim, output_len):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, n_features)
        self.output_len = output_len

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        h = h_n[-1].unsqueeze(1).repeat(1, self.output_len, 1)
        return self.fc(h)


# ===============================
# Training & Evaluation Utilities
# ===============================

def train_model(model, dataloader, epochs=20, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for x, y in dataloader:
            optimizer.zero_grad()
            preds = model(x)
            preds = preds[0] if isinstance(preds, tuple) else preds
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")


def evaluate_model(model, dataloader) -> Tuple[float, float]:
    model.eval()
    preds_all, targets_all = [], []

    with torch.no_grad():
        for x, y in dataloader:
            preds = model(x)
            preds = preds[0] if isinstance(preds, tuple) else preds
            preds_all.append(preds.numpy())
            targets_all.append(y.numpy())

    preds_all = np.concatenate(preds_all)
    targets_all = np.concatenate(targets_all)

    rmse = np.sqrt(mean_squared_error(targets_all.flatten(), preds_all.flatten()))
    mape = mean_absolute_percentage_error(targets_all.flatten(), preds_all.flatten())
    return rmse, mape


# ===============================
# Attention Visualization
# ===============================

def plot_attention_weights(attention_weights):
    """
    Visualizes attention heatmap for one sample.
    """
    # Fix: Removed .mean(axis=0) as imshow expects a 2D array for the heatmap
    weights = attention_weights[0].detach().numpy()

    plt.figure(figsize=(8, 4))
    plt.imshow(weights, aspect="auto", cmap="viridis")
    plt.colorbar(label="Attention Weight")
    plt.xlabel("Historical Time Steps")
    plt.ylabel("Query Time Step")
    plt.title("Self-Attention Weight Visualization")
    plt.tight_layout()
    plt.show()


# ===============================
# ARIMA Baseline
# ===============================

def arima_baseline(series: np.ndarray, order=(5, 1, 0)):
    model = ARIMA(series, order=order)
    model_fit = model.fit()
    return model_fit


# ===============================
# Main Execution
# ===============================

def main():
    # Hyperparameters
    INPUT_LEN = 30
    OUTPUT_LEN = 10
    HIDDEN_DIM = 64
    BATCH_SIZE = 32
    EPOCHS = 15

    # Data
    df = generate_multivariate_time_series()
    data = df.values

    train_data = data[:1200]
    test_data = data[1200:]

    train_ds = TimeSeriesDataset(train_data, INPUT_LEN, OUTPUT_LEN)
    test_ds = TimeSeriesDataset(test_data, INPUT_LEN, OUTPUT_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    # Models
    attention_model = Seq2SeqAttentionModel(
        n_features=data.shape[1],
        hidden_dim=HIDDEN_DIM,
        output_len=OUTPUT_LEN
    )

    lstm_model = StandardLSTM(
        n_features=data.shape[1],
        hidden_dim=HIDDEN_DIM,
        output_len=OUTPUT_LEN
    )

    print("\nTraining Attention-Based Seq2Seq Model")
    train_model(attention_model, train_loader, EPOCHS)

    print("\nTraining Standard LSTM Baseline")
    train_model(lstm_model, train_loader, EPOCHS)

    # Evaluation
    att_rmse, att_mape = evaluate_model(attention_model, test_loader)
    lstm_rmse, lstm_mape = evaluate_model(lstm_model, test_loader)

    print("\nEvaluation Results")
    print(f"Attention Model  RMSE: {att_rmse:.4f}, MAPE: {att_mape:.4f}")
    print(f"Standard LSTM   RMSE: {lstm_rmse:.4f}, MAPE: {lstm_mape:.4f}")

    # Attention Visualization
    x_sample, _ = next(iter(test_loader))
    _, att_weights = attention_model(x_sample)
    plot_attention_weights(att_weights)

    # ARIMA baseline (single feature)
    arima_model = arima_baseline(train_data[:, 0])
    forecast = arima_model.forecast(steps=len(test_data))
    rmse_arima = np.sqrt(mean_squared_error(test_data[:, 0], forecast))
    print(f"ARIMA Baseline RMSE (feature_0): {rmse_arima:.4f}")


if __name__ == "__main__":
    main()

   



