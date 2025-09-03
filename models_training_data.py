# This file basically takes one stock, gets its data, passes it through pre trained models, and all information possible
# (extra features, price predictions, labels made from price predictions, extra computed features, into an excel file)
# I believe that multiple stocks data have been updated into this excel file, not sure how this excel file will be used


import math
import yfinance as yf
import numpy as np
import torch
import torch.nn as nn
import joblib
import pandas as pd
import os

#-----------------------------------------#
class ResNetLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=128, num_layers=2, dropout=0.4, extra_feature_dim=8):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            dropout=dropout, batch_first=True, bidirectional=True)
        self.res_fc1 = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.res_fc2 = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.activation = nn.ReLU()
        self.bn = nn.BatchNorm1d(hidden_dim * 2 + extra_feature_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2 + extra_feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # raw logits
        )

    def forward(self, x, extra_features):
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]
        residual = self.activation(self.res_fc1(out))
        residual = self.res_fc2(residual)
        out = out + residual
        out = torch.cat([out, extra_features], dim=1)
        out = self.bn(out)
        out = self.dropout(out)
        return self.fc(out).squeeze()
# -----------------------------------------#
# -----------------------------------------#
def ema(series, span):
    series = np.asarray(series, dtype=np.float32).flatten()  # ensure 1D float32 array
    ema_vals = np.zeros(len(series), dtype=np.float32)
    alpha = 2 / (span + 1)

    ema_vals[0] = series[0]
    for i in range(1, len(series)):
        ema_vals[i] = alpha * series[i] + (1 - alpha) * ema_vals[i - 1]

    return ema_vals

def compute_macd_histogram(prices, short=58, long=125, signal=44):
    if len(prices) < long + signal:
        return 0.0
    ema_short = ema(prices, short)
    ema_long = ema(prices, long)
    macd_line = ema_short - ema_long
    signal_line = ema(macd_line, signal)
    return float(macd_line[-1] - signal_line[-1])  # ✅ Cast to float

def compute_bollinger_band_width(prices, period, k=2):
    if len(prices) < period:
        return 0.0
    sma = np.mean(prices[-period:])
    std = np.std(prices[-period:])
    return (2 * k * std) / (sma + 1e-8)

def compute_extra_features(prices):
    window = np.array(prices, dtype=np.float32)
    up_down_ratio = np.sum(np.diff(window) > 0) / (np.sum(np.diff(window) < 0) + 1e-8)
    drift = (window[-1] - window[0]) / len(window)
    volatility = np.std(window)

    # RSI
    gain = np.sum(np.clip(np.diff(window), 0, None))
    loss = np.sum(np.abs(np.clip(np.diff(window), None, 0)))
    avg_gain = gain / (len(window) - 1)
    avg_loss = loss / (len(window) - 1)
    RS = avg_gain / (avg_loss + 1e-6)
    RSI = 100 - (100 / (1 + RS))

    macd_hist = compute_macd_histogram(window)
    macd_hist_norm = macd_hist / (window[-1] + 1e-8)
    bbw_30 = compute_bollinger_band_width(window, 30)
    bbw_100 = compute_bollinger_band_width(window, 100)
    bbw_200 = compute_bollinger_band_width(window, 200)

    return [
        float(up_down_ratio),
        float(drift),
        float(volatility),
        float(RSI),
        float(macd_hist_norm),
        float(bbw_30),
        float(bbw_100),
        float(bbw_200)
    ]
# -----------------------------------------#
# -----------------------------------------#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize all models
Timmy_10 = ResNetLSTM().to(device)
Jason_30 = ResNetLSTM().to(device)
Fedra_60 = ResNetLSTM().to(device)
Larry_150 = ResNetLSTM().to(device)

# Load weights
Timmy_10.load_state_dict(torch.load("Timmy_10.pth", map_location=device))
Jason_30.load_state_dict(torch.load("Jason_30.pth", map_location=device))
Fedra_60.load_state_dict(torch.load("Fedra_60.pth", map_location=device))
Larry_150.load_state_dict(torch.load("Larry_150.pth", map_location=device))

# Set to eval mode
Timmy_10.eval()
Jason_30.eval()
Fedra_60.eval()
Larry_150.eval()

Timmy_10_scaler = joblib.load("Timmy_10_scaler.pkl")
Jason_30_scaler = joblib.load("Jason_30_scaler.pkl")
Fedra_60_scaler = joblib.load("Fedra_60_scaler.pkl")
Larry_150_scaler = joblib.load("Larry_150_scaler.pkl")
# -----------------------------------------#

# 50, 130, 250, 900 window sizes

largest_window_size_needed = 900
largest_horizon = 150
Timmy_10_window_size = 50
Jason_30_window_size = 130
Fedra_60_window_size = 250
Larry_150_window_size = 900


ticker = "INTC"
start_date = "1995-01-01"
end_date = "2025-01-01"
data = yf.download(ticker, start=start_date, end=end_date)
closing_prices = data["Close"].values
closing_prices = np.array(closing_prices).flatten()

log_prices = np.log(closing_prices + 1e-8)
log_returns = np.diff(log_prices)

meta_records = []

for i in range(len(log_returns) - largest_window_size_needed - largest_horizon):

    current_price = closing_prices[i + largest_window_size_needed]

    future_price_10 = closing_prices[i + largest_window_size_needed + 10]
    future_price_30 = closing_prices[i + largest_window_size_needed + 30]
    future_price_60 = closing_prices[i + largest_window_size_needed + 60]
    future_price_90 = closing_prices[i + largest_window_size_needed + 90]
    future_price_150 = closing_prices[i + largest_window_size_needed + 150]

    price_window_timmy_10 = closing_prices[i : i + largest_window_size_needed][-Timmy_10_window_size:]
    price_window_jason_30 = closing_prices[i: i + largest_window_size_needed][-Jason_30_window_size:]
    price_window_fedra_60 = closing_prices[i: i + largest_window_size_needed][-Fedra_60_window_size:]
    price_window_larry_150 = closing_prices[i: i + largest_window_size_needed][-Larry_150_window_size:]

    largest_window = log_returns[i : i + largest_window_size_needed]

    window_timmy_10 = largest_window[-Timmy_10_window_size:]
    window_jason_30 = largest_window[-Jason_30_window_size:]
    window_fedra_60 = largest_window[-Fedra_60_window_size:]
    window_larry_150 = largest_window[-Larry_150_window_size:]

    min_val_timmy_10 = window_timmy_10.min()
    min_val_jason_30 = window_jason_30.min()
    min_val_fedra_60 = window_fedra_60.min()
    min_val_larry_150 = window_larry_150.min()

    max_val_timmy_10 = window_timmy_10.max()
    max_val_jason_30 = window_jason_30.max()
    max_val_fedra_60 = window_fedra_60.max()
    max_val_larry_150 = window_larry_150.max()

    if min_val_timmy_10 == max_val_timmy_10 or min_val_jason_30 == max_val_jason_30 or min_val_fedra_60 == max_val_fedra_60 or min_val_larry_150 == max_val_larry_150:
        continue

    window_norm_timmy_10 = (window_timmy_10 - min_val_timmy_10) / (max_val_timmy_10 - min_val_timmy_10)
    window_norm_jason_30 = (window_jason_30 - min_val_jason_30) / (max_val_jason_30 - min_val_jason_30)
    window_norm_fedra_60 = (window_fedra_60 - min_val_fedra_60) / (max_val_fedra_60 - min_val_fedra_60)
    window_norm_larry_150 = (window_larry_150 - min_val_larry_150) / (max_val_larry_150 - min_val_larry_150)

    window_tensor_timmy_10 = torch.tensor(window_norm_timmy_10, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
    window_tensor_jason_30 = torch.tensor(window_norm_jason_30, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
    window_tensor_fedra_60 = torch.tensor(window_norm_fedra_60, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
    window_tensor_larry_150 = torch.tensor(window_norm_larry_150, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)

    features_timmy_10 = compute_extra_features(price_window_timmy_10)
    features_jason_30 = compute_extra_features(price_window_jason_30)
    features_fedra_60 = compute_extra_features(price_window_fedra_60)
    features_larry_150 = compute_extra_features(price_window_larry_150)

    features_timmy_10_copy = features_timmy_10
    features_jason_30_copy = features_jason_30
    features_fedra_60_copy = features_fedra_60
    features_larry_150_copy = features_larry_150

    features_timmy_10 = Timmy_10_scaler.transform(np.array(features_timmy_10).reshape(1, -1))
    features_jason_30 = Jason_30_scaler.transform(np.array(features_jason_30).reshape(1, -1))
    features_fedra_60 = Fedra_60_scaler.transform(np.array(features_fedra_60).reshape(1, -1))
    features_larry_150 = Larry_150_scaler.transform(np.array(features_larry_150).reshape(1, -1))

    features_tensor_timmy_10 = torch.tensor(features_timmy_10, dtype=torch.float32).to(device)
    features_tensor_jason_30 = torch.tensor(features_jason_30, dtype=torch.float32).to(device)
    features_tensor_fedra_60 = torch.tensor(features_fedra_60, dtype=torch.float32).to(device)
    features_tensor_larry_150 = torch.tensor(features_larry_150, dtype=torch.float32).to(device)

    with torch.no_grad():
        prob_timmy = torch.sigmoid(Timmy_10(window_tensor_timmy_10, features_tensor_timmy_10)).item()
        prob_jason = torch.sigmoid(Jason_30(window_tensor_jason_30, features_tensor_jason_30)).item()
        prob_fedra = torch.sigmoid(Fedra_60(window_tensor_fedra_60, features_tensor_fedra_60)).item()
        prob_larry = torch.sigmoid(Larry_150(window_tensor_larry_150, features_tensor_larry_150)).item()

    label_10 = 1.0 if future_price_10 > current_price else 0.0
    label_30 = 1.0 if future_price_30 > current_price else 0.0
    label_60 = 1.0 if future_price_60 > current_price else 0.0
    label_90 = 1.0 if future_price_90 > current_price else 0.0
    label_150 = 1.0 if future_price_150 > current_price else 0.0

    return_10 = (future_price_10 - current_price) / current_price
    return_30 = (future_price_30 - current_price) / current_price
    return_60 = (future_price_60 - current_price) / current_price
    return_90 = (future_price_90 - current_price) / current_price
    return_150 = (future_price_150 - current_price) / current_price

    log_return_10 = math.log(future_price_10) - math.log(current_price)
    log_return_30 = math.log(future_price_30) - math.log(current_price)
    log_return_60 = math.log(future_price_60) - math.log(current_price)
    log_return_90 = math.log(future_price_90) - math.log(current_price)
    log_return_150 = math.log(future_price_150) - math.log(current_price)

    meta_records.append({
        "prob_timmy": prob_timmy,
        "prob_jason": prob_jason,
        "prob_fedra": prob_fedra,
        "prob_larry": prob_larry,
        "label_10": label_10,
        "label_30": label_30,
        "label_60": label_60,
        "label_90": label_90,
        "label_150": label_150,
        "return_10": return_10,
        "return_30": return_30,
        "return_60": return_60,
        "return_90": return_90,
        "return_150": return_150,
        "log_return_10": log_return_10,
        "log_return_30": log_return_30,
        "log_return_60": log_return_60,
        "log_return_90": log_return_90,
        "log_return_150": log_return_150,
        "timmy_up_down_ratio": features_timmy_10_copy[0],
        "timmy_drift": features_timmy_10_copy[1],
        "timmy_volatility": features_timmy_10_copy[2],
        "timmy_RSI": features_timmy_10_copy[3],
        "timmy_macd_hist_norm": features_timmy_10_copy[4],
        "timmy_bbw_30": features_timmy_10_copy[5],
        "timmy_bbw_100": features_timmy_10_copy[6],
        "timmy_bbw_200": features_timmy_10_copy[7],
        "jason_up_down_ratio": features_jason_30_copy[0],
        "jason_drift": features_jason_30_copy[1],
        "jason_volatility": features_jason_30_copy[2],
        "jason_RSI": features_jason_30_copy[3],
        "jason_macd_hist_norm": features_jason_30_copy[4],
        "jason_bbw_30": features_jason_30_copy[5],
        "jason_bbw_100": features_jason_30_copy[6],
        "jason_bbw_200": features_jason_30_copy[7],
        "fedra_up_down_ratio": features_fedra_60_copy[0],
        "fedra_drift": features_fedra_60_copy[1],
        "fedra_volatility": features_fedra_60_copy[2],
        "fedra_RSI": features_fedra_60_copy[3],
        "fedra_macd_hist_norm": features_fedra_60_copy[4],
        "fedra_bbw_30": features_fedra_60_copy[5],
        "fedra_bbw_100": features_fedra_60_copy[6],
        "fedra_bbw_200": features_fedra_60_copy[7],
        "larry_up_down_ratio": features_larry_150_copy[0],
        "larry_drift": features_larry_150_copy[1],
        "larry_volatility": features_larry_150_copy[2],
        "larry_RSI": features_larry_150_copy[3],
        "larry_macd_hist_norm": features_larry_150_copy[4],
        "larry_bbw_30": features_larry_150_copy[5],
        "larry_bbw_100": features_larry_150_copy[6],
        "larry_bbw_200": features_larry_150_copy[7],
    })


file_path = "meta_model_training_data.xlsx"
new_df = pd.DataFrame(meta_records)

if os.path.exists(file_path):
    existing_df = pd.read_excel(file_path)
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
else:
    combined_df = new_df

combined_df.to_excel(file_path, index=False)
print(f"✅ Appended {len(new_df)} new rows to Excel. Total rows now: {len(combined_df)}")




