import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
import random
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
import joblib

# =============================================================
# Global Parameters
# =============================================================

# Size of historical lookback window (number of timesteps fed into the LSTM)
window_size = 50

# Forecast horizon: number of timesteps ahead to predict binary price direction
forecast_horizon = 10

# Maximum number of sampled training windows per stock (controls dataset size)
max_windows_per_stock = 100

# Maximum number of unique stocks to include in the dataset
max_stocks = 10000

# =============================================================
# Technical Indicator Utilities
# =============================================================

def ema(series, span):
    """
    Compute the Exponential Moving Average (EMA) of a price series.

    Args:
        series (np.ndarray): Array of stock prices.
        span (int): Span parameter controlling the smoothing factor.

    Returns:
        np.ndarray: EMA-smoothed series.
    """
    ema_vals = np.zeros_like(series)
    alpha = 2 / (span + 1)
    ema_vals[0] = series[0]
    for i in range(1, len(series)):
        ema_vals[i] = alpha * series[i] + (1 - alpha) * ema_vals[i - 1]
    return ema_vals

def compute_macd_histogram(prices, short=58, long=125, signal=44):
    """
    Compute the MACD histogram for a stock price series.

    Args:
        prices (array-like): Stock prices.
        short (int): Short-term EMA span.
        long (int): Long-term EMA span.
        signal (int): Signal line EMA span.

    Returns:
        float: Most recent MACD histogram value.
    """
    prices = np.array(prices, dtype=np.float32)
    if len(prices) < long + signal:
        return 0.0  # Not enough data to compute MACD

    ema_short = ema(prices, span=short)
    ema_long = ema(prices, span=long)
    macd_line = ema_short - ema_long

    signal_line = ema(macd_line, span=signal)
    macd_histogram = macd_line - signal_line

    return macd_histogram[-1]

def compute_bollinger_band_width(prices, period, k=2):
    """
    Compute normalized Bollinger Band Width (BBW).

    Args:
        prices (array-like): Stock prices.
        period (int): Lookback window size.
        k (float): Number of standard deviations for band width.

    Returns:
        float: Normalized band width value.
    """
    prices = np.array(prices, dtype=np.float32)
    if len(prices) < period:
        return 0.0

    sma = np.mean(prices[-period:])
    std = np.std(prices[-period:])

    upper_band = sma + k * std
    lower_band = sma - k * std

    bbw = (upper_band - lower_band) / (sma + 1e-8)
    return bbw

def find_optimal_threshold(y_true, y_probs):
    """
    Determine optimal decision threshold using Youden's J statistic.

    Args:
        y_true (list): Ground-truth binary labels.
        y_probs (list): Predicted probabilities.

    Returns:
        tuple: (optimal_threshold, fpr, tpr, auc_score)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    auc_score = auc(fpr, tpr)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold, fpr, tpr, auc_score

# =============================================================
# Data Preprocessing
# =============================================================

def process_stock_data(data, window_size, forecast_horizon, max_windows_per_stock, max_stocks=None):
    """
    Transform raw stock price series into supervised learning samples.

    Each sample consists of:
        - Normalized log-return window (input sequence)
        - Binary label (1 = price increase, 0 = price decrease)
        - Normalization scalars (min, max)
        - Engineered technical indicators (RSI, MACD, BBW, etc.)

    Args:
        data (list): List of stock price histories.
        window_size (int): Number of timesteps per input window.
        forecast_horizon (int): Prediction horizon.
        max_windows_per_stock (int): Max number of windows per stock.
        max_stocks (int, optional): Limit on number of stocks processed.

    Returns:
        tuple: (windows, labels, norm_mins, norm_maxs, extra_features)
    """

    windows, labels, norm_mins, norm_maxs, extra_features = [], [], [], [], []

    # Randomly sample subset of stocks if dataset is too large
    if max_stocks and len(data) > max_stocks:
        data = random.sample(list(data), max_stocks)

    for stock_prices in data:
        stock_prices = np.array(stock_prices, dtype=np.float32)

        # Ensure sufficient data length
        if len(stock_prices) < window_size + forecast_horizon + 1:
            continue

        log_prices = np.log(stock_prices + 1e-8)
        log_returns = np.diff(log_prices)

        max_start = len(log_returns) - window_size - forecast_horizon
        if max_start <= 0:
            continue

        # Sample non-overlapping(ish) starting indices
        starts = random.sample(
            range(0, max_start, 10),
            min(max_windows_per_stock, max_start, len(range(0, max_start, 10)))
        )

        for i in starts:
            # --- Input Window (normalized log returns) ---
            window = log_returns[i:i + window_size]
            min_val, max_val = window.min(), window.max()
            if max_val - min_val == 0:
                continue
            window_norm = (window - min_val) / (max_val - min_val)

            # --- Binary Label (future direction) ---
            price_now = stock_prices[i + window_size]
            future_price = stock_prices[i + window_size + forecast_horizon]
            label = 1.0 if future_price > price_now else 0.0

            # --- Extra Features (technical indicators) ---
            price_window = stock_prices[i:i + window_size + 1]
            up_down_ratio = np.sum(np.diff(price_window) > 0) / (np.sum(np.diff(price_window) < 0) + 1e-8)
            drift = (price_window[-1] - price_window[0]) / window_size
            volatility = np.std(price_window)

            gain = np.sum(np.clip(np.diff(price_window), 0, None))
            loss = np.sum(np.abs(np.clip(np.diff(price_window), None, 0)))
            avg_gain = gain / (len(price_window) - 1)
            avg_loss = loss / (len(price_window) - 1)
            RS = avg_gain / (avg_loss + 1e-6)
            RSI = 100 - (100 / (1 + RS))

            macd_hist = compute_macd_histogram(price_window)
            macd_hist_norm = macd_hist / (price_window[-1] + 1e-8)
            bbw_30 = compute_bollinger_band_width(price_window, 30)
            bbw_100 = compute_bollinger_band_width(price_window, 100)
            bbw_200 = compute_bollinger_band_width(price_window, 200)

            feature_vector = [
                up_down_ratio,
                drift,
                volatility,
                RSI,
                macd_hist_norm,
                bbw_30,
                bbw_100,
                bbw_200
            ]

            # Append processed sample
            windows.append(window_norm)
            labels.append(label)
            norm_mins.append(min_val)
            norm_maxs.append(max_val)
            extra_features.append(feature_vector)

    # Convert to arrays
    windows = np.array(windows, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)
    norm_mins = np.array(norm_mins, dtype=np.float32)
    norm_maxs = np.array(norm_maxs, dtype=np.float32)
    extra_features = np.array(extra_features, dtype=np.float32)

    # Normalize engineered features
    scaler = StandardScaler()
    extra_features = scaler.fit_transform(extra_features)
    joblib.dump(scaler, "Timmy_10_scaler.pkl")

    return windows, labels, norm_mins, norm_maxs, extra_features

# =============================================================
# Dataset Wrapper
# =============================================================

class StockDataset(Dataset):
    """
    Custom PyTorch Dataset for stock prediction.
    Each item returns:
        - Window of normalized log returns (sequence)
        - Binary label (price up/down)
        - Normalization scalars (min, max)
        - Extra technical features
    """

    def __init__(self, windows, labels, norm_mins, norm_maxs, extra_features):
        self.windows = torch.tensor(windows, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.norm_mins = torch.tensor(norm_mins, dtype=torch.float32)
        self.norm_maxs = torch.tensor(norm_maxs, dtype=torch.float32)
        self.extra_features = torch.tensor(extra_features, dtype=torch.float32)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        return (
            self.windows[index].unsqueeze(-1),
            self.labels[index],
            self.norm_mins[index],
            self.norm_maxs[index],
            self.extra_features[index]
        )

# =============================================================
# Data Loading
# =============================================================

train_data = np.load("/Users/aadityamalhotra/Documents/Quantitative Finance/training_data.npy", allow_pickle=True)
train_windows, train_labels, train_mins, train_maxs, train_extra_feat = process_stock_data(
    train_data, window_size, forecast_horizon, max_windows_per_stock, max_stocks
)
train_dataset_full = StockDataset(train_windows, train_labels, train_mins, train_maxs, train_extra_feat)

# Train-validation split
total_len = len(train_dataset_full)
train_len = int(0.7 * total_len)
val_len = total_len - train_len
train_dataset, val_dataset = random_split(train_dataset_full, [train_len, val_len])

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)


# --------------------------------------------------------------
#                       LOAD TEST DATA
# --------------------------------------------------------------

# Load testing stock price data from .npy file
test_data = np.load("/Users/aadityamalhotra/Documents/Quantitative Finance/testing_data.npy", allow_pickle=True)

# Process the test data into windows, labels, normalized min/max, and extra features
test_windows, test_labels, test_mins, test_maxs, test_extra_feat = process_stock_data(
    test_data,
    window_size,
    forecast_horizon,
    max_windows_per_stock
)

# Wrap processed test data in a PyTorch Dataset
test_dataset = StockDataset(
    test_windows,
    test_labels,
    test_mins,
    test_maxs,
    test_extra_feat
)

# Create DataLoader for batch processing during evaluation
test_loader = DataLoader(test_dataset, batch_size=64)

# --------------------------------------------------------------
#                       LSTM MODEL ARCHITECTURE
# --------------------------------------------------------------
class ResNetLSTM(nn.Module):
    """
    Combined ResNet + Bi-directional LSTM architecture for stock price movement prediction.
    Incorporates extra engineered features along with LSTM output.
    """

    def __init__(self, input_dim=1, hidden_dim=128, num_layers=2, dropout=0.4, extra_feature_dim=8):
        super().__init__()
        # Bidirectional LSTM for sequential modeling of log-return windows
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )

        # Residual fully-connected layers applied after LSTM output
        self.res_fc1 = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.res_fc2 = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.activation = nn.ReLU()

        # BatchNorm after concatenating LSTM output with extra features
        self.bn = nn.BatchNorm1d(hidden_dim * 2 + extra_feature_dim)
        self.dropout = nn.Dropout(dropout)

        # Final fully-connected network producing single logit output
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2 + extra_feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # raw logits for binary classification
        )

    def forward(self, x, extra_features):
        """
        Forward pass:
        1. Pass windowed sequence through Bi-LSTM
        2. Apply residual connection via two FC layers
        3. Concatenate with extra engineered features
        4. Apply BatchNorm and Dropout
        5. Produce final logit output
        """
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]  # Take last timestep output

        # Residual connection
        residual = self.activation(self.res_fc1(out))
        residual = self.res_fc2(residual)
        out = out + residual

        # Concatenate extra features
        out = torch.cat([out, extra_features], dim=1)

        # BatchNorm + Dropout
        out = self.bn(out)
        out = self.dropout(out)

        # Final output logit
        return self.fc(out).squeeze()


# --------------------------------------------------------------
#                       SETUP TRAINING
# --------------------------------------------------------------

# Determine device: MPS (Apple), CUDA (GPU), or CPU fallback
device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

# Instantiate model and move to device
model = ResNetLSTM().to(device)

# Binary classification loss function
criterion = nn.BCEWithLogitsLoss()

# Adam optimizer with L2 weight decay
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# Learning rate scheduler: decrease LR every 2 epochs by gamma factor
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.75)

# --------------------------------------------------------------
#                       TRAINING LOOP
# --------------------------------------------------------------

print("Training started 🤲🏻")

best_val_loss = float('inf')  # Track best validation loss for early stopping
patience = 3                  # Epochs to wait for improvement
epochs_no_improve = 0
num_epochs = 10
optimal_threshold = 0.5       # Default threshold for binary classification

for epoch in range(num_epochs):
    model.train()             # Set model to training mode
    train_loss = 0

    for X_batch, y_batch, _, _, extra_feat in train_loader:
        # Move tensors to device
        X_batch, y_batch, extra_feat = X_batch.to(device), y_batch.to(device), extra_feat.to(device)

        optimizer.zero_grad()   # Reset gradients

        # Forward pass
        logits = model(X_batch, extra_feat)

        # Confidence-based regularization: encourage outputs away from 0.5
        confidence = torch.abs(torch.sigmoid(logits) - 0.5)
        confidence_loss = torch.mean(1 - confidence)

        # Total loss = BCE + confidence penalty
        loss = criterion(logits, y_batch) + 0.5 * confidence_loss
        loss.backward()  # Backpropagation

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()  # Update weights
        train_loss += loss.item()

    # Step learning rate scheduler
    scheduler.step()

    # ----------------------------------------------------------
    #                   VALIDATION LOOP
    # ----------------------------------------------------------
    model.eval()              # Switch to evaluation mode
    val_loss = 0
    all_val_probs = []
    all_val_labels = []

    with torch.no_grad():
        for X_batch, y_batch, _, _, extra_feat in val_loader:
            X_batch, y_batch, extra_feat = X_batch.to(device), y_batch.to(device), extra_feat.to(device)
            logits = model(X_batch, extra_feat)
            loss = criterion(logits, y_batch)
            val_loss += loss.item()

            # Collect probabilities and labels for ROC analysis
            probs = torch.sigmoid(logits).cpu().numpy()
            all_val_probs.extend(probs)
            all_val_labels.extend(y_batch.cpu().numpy())

    # Compute optimal threshold using Youden's J statistic
    optimal_threshold, fpr, tpr, auc_score = find_optimal_threshold(all_val_labels, all_val_probs)

    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    print(
        f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, "
        f"Val Loss = {avg_val_loss:.4f}, AUC = {auc_score:.4f}, "
        f"Optimal Threshold = {optimal_threshold:.4f}"
    )

    # ----------------------------------------------------------
    #              EARLY STOPPING & MODEL CHECKPOINT
    # ----------------------------------------------------------
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), "Timmy_10.pth")  # Save best model
        best_optimal_threshold = optimal_threshold       # Save optimal threshold
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("Early stopping triggered")
            break

# Use optimal threshold from best validation epoch
optimal_threshold = best_optimal_threshold
print(f"\n🎯 Using optimal threshold from best validation epoch: {optimal_threshold:.4f}")

# Load best model for testing/inference
model.load_state_dict(torch.load("Timmy_10.pth"))

# --------------------------------------------------------------
#                     TESTING AND EVALUATION
# --------------------------------------------------------------

# Set model to evaluation mode (disables dropout, batchnorm updates)
model.eval()

# Initialize counters for accuracy and average percentage error
total = 0
correct = 0
total_ape = 0

# Define thresholds for top/bottom prediction analysis
up_hit_rate_thres = 0.7
low_hit_rate_thres = 0.3

# Counters for top/bottom hits and misses
upp_hit = 0
upp_miss = 0
low_hit = 0
low_miss = 0

print("Testing started 🤲🏻")

# Lists for batch-wise statistics (mean, std, lengths)
total_mean = 0
means = []
total_std = 0
stds = []
lengths = []

# --------------------------------------------------------------
#                     BATCH-WISE EVALUATION
# --------------------------------------------------------------
with torch.no_grad():
    for X_batch, y_batch, min_batch, max_batch, extra_feat in test_loader:
        # Move inputs and extra features to computation device
        X_batch, extra_feat = X_batch.to(device), extra_feat.to(device)

        # Forward pass through the model
        logits = model(X_batch, extra_feat).cpu()

        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)

        # Generate predictions using previously computed optimal threshold
        preds = (probs >= optimal_threshold).float()

        # Track batch statistics
        means.append(torch.mean(probs))
        stds.append(torch.std(probs))
        lengths.append(len(probs))

        # ----------------------------------------------------------
        # Top-K analysis: evaluate the 10 most confident positive predictions
        # ----------------------------------------------------------
        topk_vals, topk_indices = torch.topk(probs, 10)
        for i in topk_indices:
            i = int(i)
            if y_batch[i] == 1:
                upp_hit += 1  # Correct top prediction
            else:
                upp_miss += 1  # Incorrect top prediction

        # ----------------------------------------------------------
        # Bottom-K analysis: evaluate the 10 least confident predictions
        # ----------------------------------------------------------
        bottomk_vals, bottomk_indices = torch.topk(probs, 10, largest=False)
        for i in bottomk_indices:
            i = int(i)
            if y_batch[i] == 0:
                low_hit += 1  # Correct bottom prediction
            else:
                low_miss += 1  # Incorrect bottom prediction

        # ----------------------------------------------------------
        # Overall accuracy calculation
        # ----------------------------------------------------------
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

        # ----------------------------------------------------------
        # Optional: compute Average Percentage Error (APE) on real prices
        # ----------------------------------------------------------
        y_pred_real = probs * (max_batch - min_batch) + min_batch
        y_batch_real = y_batch * (max_batch - min_batch) + min_batch
        ape = torch.abs((y_pred_real - y_batch_real) / (y_batch_real + 1e-8)) * 100
        total_ape += ape.sum().item()

# --------------------------------------------------------------
#                       TEST EVALUATION
# --------------------------------------------------------------

# Compute overall directional classification accuracy
accuracy = correct / total

# Compute average absolute percentage error (APE) on real prices
avg_ape = total_ape / total

# Compute pooled standard deviation across batches
numerator = sum((n - 1) * (s ** 2) for s, n in zip(stds, lengths))
denominator = sum(n - 1 for n in lengths)
total_std = np.sqrt(numerator / denominator)

# Compute mean probability across all test predictions
total_mean = sum(means) / len(means)
print("MEAN PROB =", total_mean)

# Compute probability distribution percentiles for analysis
q0 = np.percentile(means, 0)    # Minimum
q1 = np.percentile(means, 25)   # First quartile (Q1)
q2 = np.percentile(means, 50)   # Median (Q2)
q3 = np.percentile(means, 75)   # Third quartile (Q3)
q4 = np.percentile(means, 100)  # Maximum

# Print percentile summary
print(f"Q0 (min)     = {q0}")
print(f"Q1 (25%)     = {q1}")
print(f"Q2 (median)  = {q2}")
print(f"Q3 (75%)     = {q3}")
print(f"Q4 (max)     = {q4}")

# Print overall standard deviation of probabilities
print("STD PROB =", total_std)

# Print classification accuracy and APE with optimal threshold
print(f"✅ Classification Accuracy (directional) with optimal threshold ({optimal_threshold:.4f}): {accuracy:.4f}")
print(f"📊 Average Absolute Percentage Error (real prices): {avg_ape:.2f}%")

# --------------------------------------------------------------
#                       MODEL PARAMETERS SUMMARY
# --------------------------------------------------------------
print("\n🔍 Model parameters:")
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")

# --------------------------------------------------------------
#                       UPPER & LOWER HIT RATE ANALYSIS
# --------------------------------------------------------------
print('-----------------------------------------------')

# Upper hits/misses: Top predicted probabilities vs true upward movement
print("Upper Hits =", upp_hit)
print("Upper Misses =", upp_miss)
print("Upper Hit Accuracy =", upp_hit / (upp_hit + upp_miss))

# Lower hits/misses: Bottom predicted probabilities vs true downward movement
print("Lower Hits =", low_hit)
print("Lower Misses =", low_miss)
print("Lower Hit Accuracy =", low_hit / (low_hit + low_miss))

print('-----------------------------------------------')