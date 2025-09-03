######################################################## FILE HEADER ########################################################
# This file preprosseses and saves data for faster training and testing purposes.
# This file calculates 5 probabilities (0-1) for 5 different time steps in the future (10, 30, 60, 90, 150) on whether the
# stock price will go up or down in the respective time steps. The probabilites are calculated by pre-trained models

# Below are the imports required for this file
import joblib
import torch.nn as nn
import yfinance as yf
import pandas as pd
import numpy as np
import pickle
import torch
from sklearn.preprocessing import StandardScaler
import multiprocessing

#
def ema(series, span):
    series = np.asarray(series, dtype=np.float32).flatten()
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
    return float(macd_line[-1] - signal_line[-1])

def compute_bollinger_band_width(prices, period, k=2):
    if len(prices) < period:
        return 0.0
    sma = np.mean(prices[-period:])
    std = np.std(prices[-period:])
    return (2 * k * std) / (sma + 1e-8)


TECH_TICKERS = {
    "AMZN": "Amazon.com, Inc.",
    "ADSK": "Autodesk, Inc.",
    "ARM": "Arm Holdings plc",
    "ASML": "ASML Holding N.V.",
    "ABNB": "Airbnb, Inc.",
    "CDW": "CDW Corporation",
    "CHTR": "Charter Communications, Inc.",
    "INTU": "Intuit Inc.",
    "PLTR": "Palantir Technologies Inc.",
    "QCOM": "QUALCOMM Incorporated",
    "PYPL": "PayPal Holdings, Inc.",
    "AMAT": "Applied Materials, Inc.",
    "CTSH": "Cognizant Technology Solutions Corporation",
    "IBM": "International Business Machines Corporation",
    "TSLA": "Tesla, Inc.",
    "ADP": "Automatic Data Processing, Inc.",
    "APP": "AppLovin Corporation",
    "SNPS": "Synopsys, Inc.",
    "STX": "Seagate Technology Holdings PLC",
    "ZBRA": "Zebra Technologies Corporation"
}
'''
TECH_TICKERS = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "GOOGL": "Alphabet Inc.",
    "META": "Meta Platforms, Inc.",           # Formerly Facebook
    "NVDA": "NVIDIA Corporation",
    "AMD": "Advanced Micro Devices, Inc.",
    "INTC": "Intel Corporation",
    "CSCO": "Cisco Systems, Inc.",
    "ORCL": "Oracle Corporation",
    "SAP": "SAP SE",
    "CRM": "Salesforce, Inc.",
    "ADBE": "Adobe Inc.",
    "AVGO": "Broadcom Inc.",
    "TXN": "Texas Instruments Incorporated",
    "ANET": "Arista Networks, Inc.",
    "NOW": "ServiceNow, Inc.",
    "NFLX": "Netflix, Inc.",
    "SHOP": "Shopify Inc.",
    "UBER": "Uber Technologies, Inc.",
    "PANW": "Palo Alto Networks, Inc.",
    "CDNS": "Cadence Design Systems, Inc.",
    "ADI": "Analog Devices, Inc.",
    "WDAY": "Workday, Inc.",
    "ZM": "Zoom Video Communications, Inc.",
    "SNOW": "Snowflake Inc.",
    "TEAM": "Atlassian Corporation Plc",
    "DOCU": "DocuSign, Inc.",
    "OKTA": "Okta, Inc.",
    "CRWD": "CrowdStrike Holdings, Inc.",
    "DDOG": "Datadog, Inc.",
    "ZS": "Zscaler, Inc.",
    "NET": "Cloudflare, Inc.",
    "SQ": "Block, Inc.",                       # Formerly Square
    "ROKU": "Roku, Inc.",
    "TWLO": "Twilio Inc."
}
'''

START_DATE = "2013-01-01"
END_DATE = "2025-01-01"
SAVE_PATH = "TECH_TESTING.pkl"
OFFSET = 900  # number of steps to offset to ensure all predictions align

class ResNetLSTM(nn.Module):

    def __init__(self, input_dim=1, hidden_dim=128, num_layers=2, dropout=0.3, extra_feature_dim=8):
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
        float(drift[0]),
        float(volatility),
        float(RSI),
        float(macd_hist_norm[0]),
        float(bbw_30),
        float(bbw_100),
        float(bbw_200)
    ]


def get_model_predictions(price_window, log_returns, ticker):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Timmy_10 = ResNetLSTM().to(device)
    Jason_30 = ResNetLSTM().to(device)
    Fedra_60 = ResNetLSTM().to(device)
    Larry_150 = ResNetLSTM().to(device)

    Timmy_10.load_state_dict(torch.load('/Users/aadityamalhotra/Documents/Quantitative Finance/hw5/prediction_models/Timmy_10.pth', map_location=device))
    Jason_30.load_state_dict(torch.load('/Users/aadityamalhotra/Documents/Quantitative Finance/hw5/prediction_models/Jason_30.pth', map_location=device))
    Fedra_60.load_state_dict(torch.load('/Users/aadityamalhotra/Documents/Quantitative Finance/hw5/prediction_models/Fedra_60.pth', map_location=device))
    Larry_150.load_state_dict(torch.load('/Users/aadityamalhotra/Documents/Quantitative Finance/hw5/prediction_models/Larry_150.pth', map_location=device))

    Timmy_10.eval()
    Jason_30.eval()
    Fedra_60.eval()
    Larry_150.eval()

    Timmy_10_scaler = joblib.load('/Users/aadityamalhotra/Documents/Quantitative Finance/hw5/scalers/Timmy_10_scaler.pkl')
    Jason_30_scaler = joblib.load('/Users/aadityamalhotra/Documents/Quantitative Finance/hw5/scalers/Jason_30_scaler.pkl')
    Fedra_60_scaler = joblib.load('/Users/aadityamalhotra/Documents/Quantitative Finance/hw5/scalers/Fedra_60_scaler.pkl')
    Larry_150_scaler = joblib.load('/Users/aadityamalhotra/Documents/Quantitative Finance/hw5/scalers/Larry_150_scaler.pkl')

    Tiberius_10 = joblib.load('/Users/aadityamalhotra/Documents/Quantitative Finance/hw5/consolidated_buy_models/consolidated_10.pkl')
    Jasorian_30 = joblib.load('/Users/aadityamalhotra/Documents/Quantitative Finance/hw5/consolidated_buy_models/consolidated_30.pkl')
    Fedralyn_60 = joblib.load('/Users/aadityamalhotra/Documents/Quantitative Finance/hw5/consolidated_buy_models/consolidated_60.pkl')
    Nancielle_90 = joblib.load('/Users/aadityamalhotra/Documents/Quantitative Finance/hw5/consolidated_buy_models/consolidated_90.pkl')
    Larethian_150 = joblib.load('/Users/aadityamalhotra/Documents/Quantitative Finance/hw5/consolidated_buy_models/consolidated_150.pkl')

    largest_window_size_needed = 900
    largest_horizon = 150

    Timmy_10_window_size = 50
    Jason_30_window_size = 130
    Fedra_60_window_size = 250
    Larry_150_window_size = 900

    all_model_probs = []

    for i in range(len(log_returns) - largest_window_size_needed):

        if i % 10 == 0:
            print(f"PROGRESS = {i * 100 / (len(log_returns) - largest_window_size_needed)}%")

        largest_window = log_returns[i: i + largest_window_size_needed]

        price_window_timmy_10 = price_window[i: i + largest_window_size_needed][-Timmy_10_window_size:]
        price_window_jason_30 = price_window[i: i + largest_window_size_needed][-Jason_30_window_size:]
        price_window_fedra_60 = price_window[i: i + largest_window_size_needed][-Fedra_60_window_size:]
        price_window_larry_150 = price_window[i: i + largest_window_size_needed][-Larry_150_window_size:]

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

        '''
        print(np.unique(window_timmy_10), len(window_timmy_10))
        print(window_timmy_10[:10])
        print("TICKER =", ticker)
        '''

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

        meta_features = pd.DataFrame([{
            "prob_timmy": prob_timmy,
            "prob_jason": prob_jason,
            "prob_fedra": prob_fedra,
            "prob_larry": prob_larry,
            "timmy_up_down_ratio": features_timmy_10_copy[0],
            "timmy_drift": features_timmy_10_copy[1],
            "timmy_volatility": features_timmy_10_copy[2],
            "timmy_RSI": features_timmy_10_copy[3],
            "timmy_bbw_30": features_timmy_10_copy[5],
            "jason_up_down_ratio": features_jason_30_copy[0],
            "jason_drift": features_jason_30_copy[1],
            "jason_volatility": features_jason_30_copy[2],
            "jason_RSI": features_jason_30_copy[3],
            "jason_bbw_30": features_jason_30_copy[5],
            "jason_bbw_100": features_jason_30_copy[6],
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
            "larry_bbw_200": features_larry_150_copy[7]
        }])

        proba_buy_10 = Tiberius_10.predict_proba(meta_features)[:, 1][0]
        proba_buy_30 = Jasorian_30.predict_proba(meta_features)[:, 1][0]
        proba_buy_60 = Fedralyn_60.predict_proba(meta_features)[:, 1][0]
        proba_buy_90 = Nancielle_90.predict_proba(meta_features)[:, 1][0]
        proba_buy_150 = Larethian_150.predict_proba(meta_features)[:, 1][0]

        all_model_probs.append([proba_buy_10, proba_buy_30, proba_buy_60, proba_buy_90, proba_buy_150])

    return np.array(all_model_probs, dtype=float)


def compute_log_returns(prices):
    prices = np.array(prices)
    prices = prices[prices > 0]  # Remove zero or negative prices
    if len(prices) < 2:
        return np.array([])
    log_prices = np.log(prices)
    return np.diff(log_prices)

def fetch_and_process(ticker):
    try:
        df = yf.download(ticker, start=START_DATE, end=END_DATE, auto_adjust=False, progress=False)
        if df.empty or 'Close' not in df.columns:
            print(f"❌ Skipping {ticker}: No 'Close' data.")
            return None

        prices = df['Close'].dropna()
        prices = prices[prices > 0]  # Remove zero or negative prices

        if len(prices) < OFFSET + 400:  # ensure enough data even after offset
            print(f"⚠️  Skipping {ticker}: Not enough price data ({len(prices)} points).")
            return None

        log_returns = compute_log_returns(prices.values)

        if len(log_returns) == 0:
            print(f"⚠️  Skipping {ticker}: Log returns too short.")
            return None

        scaler = StandardScaler()
        norm_log_returns = scaler.fit_transform(log_returns.reshape(-1, 1)).flatten()

        model_preds = get_model_predictions(prices, log_returns, ticker)

        # Slice data to drop the first OFFSET points
        raw_prices = prices.values[OFFSET + 1:]
        log_returns = log_returns[OFFSET:]
        norm_log_returns = norm_log_returns[OFFSET:]

        if len(model_preds) != len(raw_prices):
            print(f"❌ Length mismatch: model_preds={len(model_preds)} vs raw_prices={len(raw_prices)}")


        return {
            "ticker": ticker,
            "company": TECH_TICKERS[ticker],
            "raw_prices": raw_prices,
            "log_returns": log_returns,
            "norm_log_returns": norm_log_returns,
            "scaler": scaler,
            "start_date": str(prices.index[OFFSET].date()),
            "end_date": str(prices.index[-1].date()),
            "model_preds": model_preds  # Placeholder; should be filled post-model inference
        }

    # after running, program was throwing this exception
    except Exception as e:
        print(f"⚠️  Error fetching {ticker}: {e}")
        return None




def main():
    #os.makedirs("data", exist_ok=True)

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.map(fetch_and_process, list(TECH_TICKERS.keys()))

    # Filter out None results (failed fetches)
    stock_data_dict = {res['ticker']: res for res in results if res is not None}

    for ticker, data in stock_data_dict.items():
        with open(f"{ticker}_data.pkl", "wb") as f:
            pickle.dump(data, f)

    if stock_data_dict:
        with open(SAVE_PATH, "wb") as f:
            pickle.dump(stock_data_dict, f)
        print(f"\n✅ Saved data for {len(stock_data_dict)} stocks to {SAVE_PATH}")
    else:
        print("\n❌ No valid stock data saved.")


if __name__ == "__main__":
    main()