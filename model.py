# Part 1: Imports, API setup, and core functions

# 1. IMPORTS
import requests
import joblib
import numpy as np
import pandas as pd
import os
import datetime
import warnings
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Conv1D,
    MaxPooling1D,
    LSTM,
    Dropout,
    GlobalAveragePooling1D,
    LayerNormalization,
    MultiHeadAttention,
    Layer,
)

from tensorflow.keras.optimizers import Adam

# Optional imports based on availability
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    VADER_AVAILABLE = True
    vader = SentimentIntensityAnalyzer()
except ImportError:
    VADER_AVAILABLE = False

try:
    import ta  # Technical Analysis library

    TA_LIB_AVAILABLE = True
except ImportError:
    TA_LIB_AVAILABLE = False

try:
    import xgboost as xgb
    from xgboost import XGBRegressor

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from transformers import pipeline

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

warnings.filterwarnings("ignore")

# API keys (ensure these environment variables are set)
NEWSAPI_KEY = os.getenv("NEWS_API_KEY_NEWSAPI")  # NewsAPI key
FRED_KEY = os.getenv("NEWS_API_KEY_FRED")  # FRED key


# --------------------------------------------------------------------------------
# 1b) News Sentiment Analysis
# --------------------------------------------------------------------------------
def fetch_news_sentiment(ticker):
    """
    Fetch recent news headlines related to the ticker using NewsAPI and compute sentiment scores.
    Returns a DataFrame with sentiment scores over time.
    """
    news_api_key = os.getenv("NEWS_API_KEY_NEWSAPI")
    if not news_api_key:
        raise ValueError("NewsAPI key not found. Please set NEWS_API_KEY_NEWSAPI in your environment.")

    # Map ticker symbols to more meaningful search terms
    ticker_to_search_term = {
        "CC=F": "cocoa futures",
        "GC=F": "gold futures",
        "KC=F": "coffee futures",
        "NG=F": "natural gas futures",
        "^GDAXI": "DAX index",
        "^HSI": "Hang Seng Index",
        "USDJPY": "USD JPY",
        "ETHUSD": "Ethereum",
        "SOLUSD": "Solana",
        "^SPX": "S&P 500",
        "HG=F": "copper futures",
        "SI=F": "silver futures",
        "CL=F": "crude oil futures",
        "CHFJPY": "CHF JPY",
        "USDCHF": "USD CHF",
        "BNBUSD": "Binance Coin",
    }

    search_term = ticker_to_search_term.get(ticker, ticker)

    # Fetch recent headlines related to the ticker from NewsAPI
    from_date = (datetime.datetime.now() - datetime.timedelta(days=7)).strftime("%Y-%m-%d")

    params = {
        "q": search_term,
        "from": from_date,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 100,
        "apiKey": news_api_key,
    }

    news_url = "https://newsapi.org/v2/everything"

    news_response = requests.get(news_url, params=params)

    if news_response.status_code != 200:
        error_message = news_response.json().get("message", "No error message provided")
        raise ValueError(
            f"Failed to fetch news data from NewsAPI. Status code: {news_response.status_code}. "
            f"Error message: {error_message}"
        )

    news_data = news_response.json()
    articles = news_data.get("articles", [])

    if not articles:
        print(f"No news articles found for ticker {ticker}.")
        return pd.DataFrame()

    headlines = [article["title"] for article in articles]
    publishedAt = [article["publishedAt"] for article in articles]

    publishedAt = pd.to_datetime(publishedAt).tz_localize(None)
    df_news = pd.DataFrame({"headline": headlines, "publishedAt": publishedAt})
    df_news.set_index("publishedAt", inplace=True)

    if VADER_AVAILABLE:
        df_news["compound"] = df_news["headline"].apply(lambda x: vader.polarity_scores(x)["compound"])
        df_news["neg"] = df_news["headline"].apply(lambda x: vader.polarity_scores(x)["neg"])
        df_news["neu"] = df_news["headline"].apply(lambda x: vader.polarity_scores(x)["neu"])
        df_news["pos"] = df_news["headline"].apply(lambda x: vader.polarity_scores(x)["pos"])
    elif TRANSFORMERS_AVAILABLE:
        sentiment_analyzer = pipeline("sentiment-analysis")
        sentiments = sentiment_analyzer(df_news["headline"].tolist())
        df_news["sentiment"] = [s["label"] for s in sentiments]
        df_news["score"] = [s["score"] for s in sentiments]

        df_news["compound"] = df_news.apply(
            lambda x: x["score"] if x["sentiment"] == "POSITIVE" else -x["score"], axis=1
        )
        df_news["neg"] = df_news["compound"].apply(lambda x: -x if x < 0 else 0)
        df_news["neu"] = 0
        df_news["pos"] = df_news["compound"].apply(lambda x: x if x > 0 else 0)
    else:
        print("No sentiment analysis tool available. Install VADER or transformers library.")
        return pd.DataFrame()

    # Resample to align with price data, e.g., to 15-minute intervals
    df_sentiment = df_news[["compound", "neg", "neu", "pos"]].resample("15T").mean()

    return df_sentiment


# --------------------------------------------------------------------------------
# 1) Data Fetching
# --------------------------------------------------------------------------------
def fetch_live_data(tickers, retries=3):
    """
    Fetch 15-minute historical data from FinancialModelingPrep using FMP_API_KEY.
    """
    data = {}
    api_key = os.getenv("FMP_API_KEY")
    if not api_key:
        raise ValueError("API key not found. Please set FMP_API_KEY in your environment.")

    for ticker in tickers:
        for attempt in range(retries):
            try:
                ticker_api = ticker.replace('/', '')
                url = f'https://financialmodelingprep.com/api/v3/historical-chart/15min/{ticker_api}?apikey={api_key}'
                response = requests.get(url)
                response.raise_for_status()
                data_json = response.json()

                if not data_json or len(data_json) < 1:
                    continue

                df = pd.DataFrame(data_json)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                df.rename(
                    columns={
                        'close': 'Close',
                        'open': 'Open',
                        'high': 'High',
                        'low': 'Low',
                        'volume': 'Volume'
                    },
                    inplace=True
                )
                df.sort_index(inplace=True)  # ascending
                data[ticker] = df
                break
            except Exception as e:
                if attempt < retries - 1:
                    continue
                else:
                    print(f"Error fetching data for {ticker}: {e}")
        else:
            print(f"Failed to fetch data for {ticker} after {retries} retries.")
    return data


# --------------------------------------------------------------------------------
# 2) Feature Engineering
# --------------------------------------------------------------------------------
def enhance_features(df, feature_cols=None):
    """
    Enhanced feature set including moving averages, lags, log returns,
    volatility regime, advanced TA features, and news sentiment.
    Now computes only features specified in feature_cols.
    """
    if feature_cols is None:
        raise ValueError("feature_cols must be provided to enhance_features function.")
    
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    
    feature_functions = {
        'Close': lambda df: df['Close'],
        'MA_Short': lambda df: df['Close'].rolling(window=5).mean(),
        'MA_Long': lambda df: df['Close'].rolling(window=20).mean(),
        'Lag_1': lambda df: df['Close'].shift(1),
        'Lag_2': lambda df: df['Close'].shift(2),
        'Lag_3': lambda df: df['Close'].shift(3),
        'Range': lambda df: df['High'] - df['Low'],
        'Log_Return': lambda df: np.log(df['Close'] / df['Close'].shift(1)),
        'RSI': lambda df: ta.momentum.RSIIndicator(df['Close'], 14).rsi() if TA_LIB_AVAILABLE else None,
        'MACD': lambda df: ta.trend.MACD(df['Close']).macd() if TA_LIB_AVAILABLE else None,
        'MACD_Signal': lambda df: ta.trend.MACD(df['Close']).macd_signal() if TA_LIB_AVAILABLE else None,
        'MACD_Diff': lambda df: ta.trend.MACD(df['Close']).macd_diff() if TA_LIB_AVAILABLE else None,
        'Bollinger_High': lambda df: ta.volatility.BollingerBands(df['Close']).bollinger_hband() if TA_LIB_AVAILABLE else None,
        'Bollinger_Low': lambda df: ta.volatility.BollingerBands(df['Close']).bollinger_lband() if TA_LIB_AVAILABLE else None,
        'Stoch_K': lambda df: ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close']).stoch() if TA_LIB_AVAILABLE else None,
        'Stoch_D': lambda df: ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close']).stoch_signal() if TA_LIB_AVAILABLE else None,
        'Williams_%R': lambda df: ta.momentum.WilliamsRIndicator(df['High'], df['Low'], df['Close']).williams_r() if TA_LIB_AVAILABLE else None,
        'ADX': lambda df: ta.trend.ADXIndicator(df['High'], df['Low'], df['Close']).adx() if TA_LIB_AVAILABLE else None,
        'CCI': lambda df: ta.trend.CCIIndicator(df['High'], df['Low'], df['Close']).cci() if TA_LIB_AVAILABLE else None,
        'ATR': lambda df: ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range() if TA_LIB_AVAILABLE else None,
    }
    
    for feature in feature_cols:
        if feature in df.columns:
            continue
        elif feature in feature_functions:
            computation = feature_functions[feature](df)
            if computation is not None:
                df[feature] = computation
            else:
                print(f"Feature {feature} could not be computed because necessary libraries are unavailable.")
        else:
            print(f"Feature {feature} computation is not defined.")
    
    df.dropna(inplace=True)
    return df


# --------------------------------------------------------------------------------
# 3) Data Preparation
# --------------------------------------------------------------------------------
def prepare_sequence_data(df, feature_cols, time_step=60, scaler=None):
    """
    Prepare sequence data for LSTM model.
    Returns (X, y) in shape:
      X: (num_samples, time_step=60, num_features),
      y: (num_samples, 1).
    """
    data = df[feature_cols].values

    if scaler is None:
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)
    else:
        scaled_data = scaler.transform(data)

    X, y = [], []
    close_idx = feature_cols.index("Close")
    for i in range(time_step, len(scaled_data)):
        X.append(scaled_data[i - time_step : i, :])
        y.append(scaled_data[i, close_idx])
    X, y = np.array(X), np.array(y)
    return X, y, scaler

# --------------------------------------------------------------------------------
# 4) Custom Attention Layer
# --------------------------------------------------------------------------------
class AttentionLayer(Layer):
    def __init__(self, units=128):
        super().__init__()
        self.query_dense = tf.keras.layers.Dense(units, activation="relu")
        self.key_dense = tf.keras.layers.Dense(units, activation="relu")
        self.value_dense = tf.keras.layers.Dense(units, activation="relu")

    def call(self, inputs):
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        attention_weights = tf.nn.softmax(tf.matmul(query, key, transpose_b=True), axis=-1)
        weighted_sum = tf.matmul(attention_weights, value)
        return weighted_sum

# --------------------------------------------------------------------------------
# 5) Model Definitions
# --------------------------------------------------------------------------------
def build_cnn_lstm_attention_model(
    filters=64,
    kernel_size=3,
    lstm_units=100,
    dropout_rate=0.3,
    learning_rate=0.001,
    n_features=6,
):
    model = Sequential()
    model.add(Conv1D(filters, kernel_size, activation="relu", input_shape=(60, n_features)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(lstm_units, return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(AttentionLayer())
    model.add(GlobalAveragePooling1D())
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate), loss="mean_squared_error")
    return model


def build_transformer_model(
    d_model=32,
    num_heads=2,
    ff_dim=64,
    dropout_rate=0.1,
    sequence_length=60,
    n_features=6,
    learning_rate=0.001,
):
    inputs = Input(shape=(sequence_length, n_features))
    x = Dense(d_model)(inputs)
    x = Dropout(dropout_rate)(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
    x = x + attn_output
    x = LayerNormalization(epsilon=1e-6)(x)
    ff = Sequential(
        [
            Dense(ff_dim, activation="relu"),
            Dropout(dropout_rate),
            Dense(d_model),
        ]
    )
    x = x + ff(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate), loss="mean_squared_error")
    return model


def build_tcn_model(
    filters=64,
    kernel_size=3,
    dilation_rates=(1, 2, 4, 8),
    dropout_rate=0.2,
    learning_rate=0.001,
    n_features=6,
):
    input_layer = Input(shape=(60, n_features))
    x = input_layer
    for d in dilation_rates:
        x_skip = x
        x = Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=d,
            padding="causal",
            activation="relu",
        )(x)
        x = Dropout(dropout_rate)(x)
        if x_skip.shape[-1] != x.shape[-1]:
            x_skip = Conv1D(filters, 1, padding="same")(x_skip)
        x = x + x_skip
    x = GlobalAveragePooling1D()(x)
    outputs = Dense(1)(x)
    model = Model(input_layer, outputs)
    model.compile(optimizer=Adam(learning_rate), loss="mean_squared_error")
    return model


def build_informer_model(
    d_model=32,
    num_heads=2,
    dropout_rate=0.1,
    sequence_length=60,
    n_features=6,
    learning_rate=0.001,
):
    inputs = Input(shape=(sequence_length, n_features))
    x = Dense(d_model)(inputs)
    x = LayerNormalization(epsilon=1e-6)(x)
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
    x = x + attn_output
    x = LayerNormalization(epsilon=1e-6)(x)
    ff = Sequential([Dense(d_model, activation="relu"), Dense(d_model)])
    x = x + ff(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate), loss="mean_squared_error")
    return model


# --------------------------------------------------------------------------------
# 6) Ensemble Predictions with Dynamic Weights
# --------------------------------------------------------------------------------
def ensemble_predict_dynamic(models, X, y_true=None):
    preds_list = []
    weights = []
    for model_name, model in models:  # Unpack the tuple here
        preds = model.predict(X)
        preds_list.append(preds)
        if y_true is not None:
            mse = mean_squared_error(y_true, preds)
            weight = 1 / (mse + 1e-8)
            weights.append(weight)
        else:
            weights.append(1)

    preds_array = np.array(preds_list)
    weights = np.array(weights)
    normalized_weights = weights / weights.sum()
    weighted_preds = np.tensordot(normalized_weights, preds_array, axes=[0, 0])
    return weighted_preds

# --------------------------------------------------------------------------------
# 7) Trading Simulation with TP/SL and Threshold
# --------------------------------------------------------------------------------
def simulate_trades_with_sl_tp(
    predictions, actual_prices, take_profit=0.02, stop_loss=0.01, threshold=0.001
):
    if len(predictions) != len(actual_prices):
        raise ValueError("predictions vs. actual_prices size mismatch.")

    predictions = np.array(predictions)
    actual_prices = np.array(actual_prices)
    n = len(predictions)
    trades = []
    equity = [100000]
    pos = 0
    entry_price = 0

    for i in range(n - 1):
        pred_change = (predictions[i + 1] - actual_prices[i]) / actual_prices[i]
        signal = 0
        if pred_change > threshold:
            signal = 1  # BUY signal
        elif pred_change < -threshold:
            signal = -1  # SELL signal

        if pos == 0 and signal != 0:
            pos = signal
            entry_price = actual_prices[i + 1] + pos * 0.01  # Slippage
            trades.append(("ENTRY", i + 1, entry_price, pos))
        elif pos != 0:
            current_price = actual_prices[i + 1]
            change = (current_price - entry_price) / entry_price * pos
            if change >= take_profit or change <= -stop_loss or signal != pos:
                exit_price = current_price - pos * 0.01  # Slippage
                pnl = (exit_price - entry_price) * pos - 2.0  # Commission
                equity.append(equity[-1] + pnl)
                trades.append(("EXIT", i + 1, exit_price, pos))
                pos = 0
                entry_price = 0
            else:
                mtm = (current_price - entry_price) * pos
                equity.append(equity[-1] + mtm)
        else:
            equity.append(equity[-1])

    if pos != 0:
        final_price = actual_prices[-1] - pos * 0.01  # Slippage
        pnl = (final_price - entry_price) * pos - 2.0  # Commission
        equity.append(equity[-1] + pnl)
        trades.append(("EXIT_END", n - 1, final_price, pos))
        pos = 0

    return trades, equity


# --------------------------------------------------------------------------------
# 8) Performance Metrics and Backtesting Stats
# --------------------------------------------------------------------------------
def sharpe_ratio(equity, annual_factor=252 * (24 * 4)):
    returns = np.diff(equity) / equity[:-1]
    if returns.std() == 0:
        return 0
    else:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(annual_factor)
        return sharpe


def max_drawdown(equity):
    cum_max = np.maximum.accumulate(equity)
    drawdown = (equity - cum_max) / cum_max
    return drawdown.min()


def trade_statistics(trades):
    num_trades = len([t for t in trades if t[0] == "EXIT" or t[0] == "EXIT_END"])
    wins = [t for t in trades if t[0] == "EXIT" and t[3] > 0]
    losses = [t for t in trades if t[0] == "EXIT" and t[3] < 0]
    win_rate = len(wins) / num_trades if num_trades > 0 else 0
    avg_win = np.mean([t[3] for t in wins]) if wins else 0
    avg_loss = np.mean([t[3] for t in losses]) if losses else 0
    profit_factor = -avg_win / avg_loss if avg_loss != 0 else np.inf
    return {
        "Total Trades": num_trades,
        "Win Rate": win_rate,
        "Avg Win": avg_win,
        "Avg Loss": avg_loss,
        "Profit Factor": profit_factor,
    }


# --------------------------------------------------------------------------------
# 5b) XGBoost utilities
# --------------------------------------------------------------------------------
def prepare_xgb_data(df, feature_cols, horizon=1):
    if "Close" not in df.columns:
        raise ValueError("'Close' column is required as the prediction target")

    y = df["Close"].shift(-horizon)
    X = df[feature_cols]

    data = pd.concat([X, y.rename("y")], axis=1).dropna()
    X = data[feature_cols].values
    y = data["y"].values
    return X, y


def train_xgb_model(X, y, test_size=0.2, random_state=42):
    if not XGBOOST_AVAILABLE:
        raise ImportError("xgboost is not installed; pip install xgboost")

    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    model = XGBRegressor(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=random_state,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    print(f"XGBoost – MSE: {mse:.6f}, MAE: {mae:.6f}")

    return model


# --------------------------------------------------------------------------------
# 9) Alerts and Explainability
# --------------------------------------------------------------------------------
def generate_alerts(predictions, actual_prices, threshold=0.02):
    alerts = []
    for i in range(len(predictions)):
        pred_price = predictions[i]
        actual_price = actual_prices[i]
        change_pct = (pred_price - actual_price) / actual_price
        if change_pct > threshold:
            alerts.append(("BUY", i, pred_price, change_pct))
        elif change_pct < -threshold:
            alerts.append(("SELL", i, pred_price, change_pct))
    return alerts


def explain_predictions(model, X_sample, feature_names):
    try:
        import shap

        explainer = shap.KernelExplainer(model.predict, X_sample)
        shap_values = explainer.shap_values(X_sample)
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names)
    except ImportError:
        print("SHAP library not installed. Skipping explainability.")
    except Exception as e:
        print(f"Error in explainability: {e}")


# --------------------------------------------------------------------------------
# 10) Main Execution
# --------------------------------------------------------------------------------
if __name__ == "__main__":
    tickers = [
        "CC=F",
        "GC=F",
        "KC=F",
        "NG=F",
        "^GDAXI",
        "USDJPY",
        "HG=F",
        "SI=F",
        "CL=F",
        "CHFJPY",
        "USDCHF",
        "BNBUSD",
    ]

    task = "regression"

    data = fetch_live_data(tickers)
    for ticker in tickers:
        df = data.get(ticker, pd.DataFrame())
        if df.empty:
            print(f"No data for {ticker}, skipping.")
            continue

        print(f"\n--- Ticker: {ticker} ---")
        print(f"Rows: {len(df)}")

        sentiment_df = fetch_news_sentiment(ticker)

        # Define feature columns
        base_cols = [
            "Close",
            "MA_Short",
            "MA_Long",
            "Lag_1",
            "Lag_2",
            "Lag_3",
            "Range",
            "Log_Return",
        ]
        adv_cols = []
        sentiment_cols = []
        if TA_LIB_AVAILABLE:
            adv_cols = [
                "RSI",
                "MACD",
                "MACD_Signal",
                "MACD_Diff",
                "Bollinger_High",
                "Bollinger_Low",
                "Stoch_K",
                "Stoch_D",
                "Williams_%R",
                "ADX",
                "CCI",
                "ATR",
            ]
        if VADER_AVAILABLE and not sentiment_df.empty:
            sentiment_cols = ["compound", "neg", "neu", "pos"]
        feature_cols = [c for c in base_cols + adv_cols + sentiment_cols if c in df.columns]

        if len(feature_cols) == 0:
            print(f"No valid features for {ticker}.")
            continue

        forecast_horizon = 12

        df = enhance_features(df, feature_cols)
        if len(df) < 60:
            print(f"Not enough rows after feature engineering. Rows: {len(df)}")
            continue

        print(f"Enhanced Features: {feature_cols}")

        X_ts, y_ts, scaler = prepare_sequence_data(
            df, feature_cols, time_step=60
        )
        print(f"LSTM data shapes: X={X_ts.shape}, y={y_ts.shape}")

        if X_ts.size == 0 or y_ts.size == 0:
            print(f"Not enough data for LSTM preparation for {ticker}.")
            continue

        tscv = TimeSeriesSplit(n_splits=5)
        all_preds = []
        all_actuals = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X_ts)):
            X_train, X_test = X_ts[train_idx], X_ts[test_idx]
            y_train, y_test = y_ts[train_idx], y_ts[test_idx]

            print(f"\nFold {fold + 1}")

            models_to_train = [
                ('cnn_lstm', build_cnn_lstm_attention_model(n_features=len(feature_cols))),
                ('transformer', build_transformer_model(n_features=len(feature_cols))),
                ('tcn', build_tcn_model(n_features=len(feature_cols))),
                ('informer', build_informer_model(n_features=len(feature_cols)))
            ]

            trained_models = []
            for model_name, model in models_to_train:
                try:
                    model.fit(
                        X_train, y_train, 
                        epochs=5, batch_size=64, 
                        shuffle=False, verbose=0
                    )
                    trained_models.append((model_name, model))  # Append a tuple with model name and model object
                
                except (ValueError, RuntimeError) as e:
                    print(f"Error training {model_name}: {e}")
                    continue

            if not trained_models:
                print("No models were successfully trained in this fold.")
                continue

            preds = ensemble_predict_dynamic(trained_models, X_test, y_test)

            mse = mean_squared_error(y_test, preds)
            mae = mean_absolute_error(y_test, preds)
            print(f"Ensemble - MSE: {mse:.6f}, MAE: {mae:.6f}")

            # Ensure y_test is 1-dimensional
            y_test = y_test.reshape(-1)
            preds = preds.reshape(-1)
            all_preds.extend(preds)
            all_actuals.extend(y_test)

        # Convert lists to arrays for concatenation
        all_preds_flat = np.array(all_preds)
        all_actuals_flat = np.array(all_actuals)

        total_mse = mean_squared_error(all_actuals_flat, all_preds_flat)
        total_mae = mean_absolute_error(all_actuals_flat, all_preds_flat)
        print(f"\nTotal MSE: {total_mse:.6f}")
        print(f"Total MAE: {total_mae:.6f}")

        print("\n--- Simulating Trades with Stop-Loss and Take-Profit ---")

        inv_preds = []
        inv_actuals = []
        for pred, actual in zip(all_preds_flat, all_actuals_flat):
            dummy_pred = np.zeros((1, len(feature_cols)))
            dummy_pred[0, feature_cols.index("Close")] = pred
            inv_pred = scaler.inverse_transform(dummy_pred)[0, feature_cols.index("Close")]

            inv_preds.append(inv_pred)

            dummy_actual = np.zeros((1, len(feature_cols)))
            dummy_actual[0, feature_cols.index("Close")] = actual
            inv_actual = scaler.inverse_transform(dummy_actual)[0, feature_cols.index("Close")]
            inv_actuals.append(inv_actual)

        threshold = 0.001

        trades, equity = simulate_trades_with_sl_tp(
            predictions=inv_preds,
            actual_prices=inv_actuals,
            take_profit=0.02,
            stop_loss=0.01,
            threshold=threshold,
        )
        sharpe = sharpe_ratio(equity)
        mdd = max_drawdown(equity)
        stats = trade_statistics(trades)
        print(f"Sharpe Ratio: {sharpe:.2f}, Max Drawdown: {mdd:.2f}")
        print(f"Ending Equity: {equity[-1]:.2f}")
        print(f"Trade Statistics: {stats}")

        alerts = generate_alerts(inv_preds, inv_actuals, threshold=0.02)
        if alerts:
            print("\nGenerated Alerts:")
            for alert in alerts:
                print(alert)
        else:
            print("\nNo alerts generated.")

        print("\n--- Multi-step Forecast for Next 3 Days ---")
        n_steps = 3 * 24 * 4

        forecasts = []
        X_input = X_ts[-1:]

        for _ in range(n_steps):
            pred = ensemble_predict_dynamic(
                trained_models, X_input
            )

            if pred.shape != (1,):
                pred = pred.reshape(1,)

            forecasts.append(pred[0])

            last_features = X_input[0, 1:, :]
            next_feature = X_input[0, -1:, :]
            next_feature = next_feature.copy()
            next_feature[0, feature_cols.index("Close")] = pred[0]
            X_input = np.concatenate([last_features, next_feature], axis=0)
            X_input = X_input.reshape(1, X_input.shape[0], X_input.shape[1])

        inv_forecasts = []
        for pred in forecasts:
            dummy = np.zeros((1, len(feature_cols)))
            dummy[0, feature_cols.index("Close")] = pred
            inv_pred = scaler.inverse_transform(dummy)
            inv_close = inv_pred[0, feature_cols.index("Close")]
            inv_forecasts.append(inv_close)

        min_len = min(n_steps, len(inv_forecasts))
        times = pd.date_range(
            start=df.index[-1] + pd.Timedelta(minutes=15),
            periods=min_len,
            freq="15T"
        )

        df_forecast = pd.DataFrame({
            "Time": times,
            "Predicted_Price": inv_forecasts[:min_len]
        })

        df_forecast["Date"] = df_forecast["Time"].dt.date
        daily_ranges = df_forecast.groupby("Date")["Predicted_Price"].agg(["min", "max"])
        print("Predicted Price Ranges for Next 3 Days:")
        print(daily_ranges.head(3))

        print(f"\n--- Finished {ticker} ---")

        # Save models
        os.makedirs("model", exist_ok=True)
        for model_name, model in trained_models:
            model.save(f"model/{ticker}_{model_name}.h5")

        scaler_filename = f"model/{ticker}_scaler.pkl"
        joblib.dump(scaler, scaler_filename)
        print(f"Scaler saved to {scaler_filename}")

        feature_filename = f"model/{ticker}_features.pkl"
        joblib.dump(feature_cols, feature_filename)
        print(f"Feature columns saved to {feature_filename}")

        xgb_model = None
        if XGBOOST_AVAILABLE:
            print("\n--- XGBoost ---")
            X_xgb, y_xgb = prepare_xgb_data(df, feature_cols)
            xgb_model = train_xgb_model(X_xgb, y_xgb)
            joblib.dump(xgb_model, f"model/{ticker}_xgb.pkl")