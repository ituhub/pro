# Part 1: Imports, API setup, and core functions

# 1. IMPORTS
import os
import json
import joblib
import requests
import streamlit as st
import pandas as pd
import numpy as np
import nltk
import plotly.graph_objects as go
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LSTM as _KerasLSTM

class LegacyLSTM(_KerasLSTM):
    """Discard the 'time_major' argument stored in older TF-Keras .h5 files."""
    def __init__(self, *args, **kwargs):
        kwargs.pop("time_major", None)      # <- the trouble-maker
        super().__init__(*args, **kwargs)
from model import (
    AttentionLayer,
    enhance_features,
    ensemble_predict_dynamic,
    prepare_sequence_data,
    build_cnn_lstm_attention_model,
    build_transformer_model,
    build_tcn_model,
    build_informer_model,
    sharpe_ratio,
    max_drawdown,
    trade_statistics,
    generate_alerts,
)

# --- News Sentiment Functions ---

def fetch_news(ticker):
    """Fetch latest news articles using NewsAPI."""
    news_api_key = os.getenv("NEWS_API_KEY_NEWSAPI")
    if not news_api_key:
        st.error("API key for NewsAPI.org not set.")
        return []
    
    url = f'https://newsapi.org/v2/everything?q={ticker}&sortBy=publishedAt&language=en&apiKey={news_api_key}'
    try:
        r = requests.get(url)
        r.raise_for_status()
        articles = r.json().get('articles', [])
        return articles
    except Exception as e:
        st.error(f"⚠️ Could not fetch news: {e}")
        return []

def compute_sentiment_score(articles):
    """Compute average sentiment score from news articles."""
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = []

    for article in articles:
        text = ' '.join(filter(None, [article.get('title', ''), article.get('description', ''), article.get('content', '')]))
        if text:
            score = sia.polarity_scores(text)['compound']
            sentiment_scores.append(score)

    return np.mean(sentiment_scores) if sentiment_scores else 0.0

# --- Streamlit Setup ---
st.set_page_config(page_title="📈 AI Trading Predictor", layout="wide")
st.title("📊 AI-Powered Ensemble Price Predictor")
st.markdown("Uses trained models (CNN-LSTM, Transformer, TCN, Informer) for prediction.")

# --- Load Models -----------------------------------------------------
def load_all_models(ticker):
    """Load all trained models and scaler for a specific ticker."""
    try:
        custom_objs = {
            "AttentionLayer": AttentionLayer,   # your custom layer
            "LSTM": LegacyLSTM                 # 👈 NEW – use wrapper
        }
        models = {
            'cnn_lstm': load_model(f"model/{ticker}_cnn_lstm.h5",
                                   custom_objects=custom_objs),
            'transformer': load_model(f"model/{ticker}_transformer.h5",
                                      custom_objects=custom_objs),
            'tcn': load_model(f"model/{ticker}_tcn.h5",
                              custom_objects=custom_objs),
            'informer': load_model(f"model/{ticker}_informer.h5",
                                   custom_objects=custom_objs)
        }
        scaler        = joblib.load(f"model/{ticker}_scaler.pkl")
        feature_cols  = joblib.load(f"model/{ticker}_features.pkl")
        return models, scaler, feature_cols
    except Exception as e:
        st.error(f"⚠️ Model loading failed for {ticker}: {e}")
        return None, None, None
# --------------------------------------------------------------------


# --- Fetch Live Data ---
def fetch_historical_data(symbol, interval='15min'):
    """Fetch live market data from Financial Modeling Prep API"""
    key = os.getenv("FMP_API_KEY")
    if not key:
        st.error("⚠️ API key for Financial Modeling Prep is not set. Please set the 'FMP_API_KEY' environment variable.")
        return pd.DataFrame()
    url = f'https://financialmodelingprep.com/api/v3/historical-chart/{interval}/{symbol}?apikey={key}'

    try:
        r = requests.get(url)
        r.raise_for_status()
        data = r.json()

        df = pd.DataFrame(data)
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
        return df
    except Exception as e:
        st.error(f"⚠️ Could not fetch data for {symbol}: {e}")
        return pd.DataFrame()

# --- Prediction Function ---
def predict_price(df, ticker, sl_percent, tp_percent, multi_steps=0):
    """Generate price prediction and optional multi-step forecast."""
    models, scaler, feature_cols = load_all_models(ticker)
    if not models or not scaler or not feature_cols:
        return None

    try:
        df_features = enhance_features(df, feature_cols)
        if df_features is None or len(df_features) < 60:
            st.warning(f"⚠️ Not enough data for {ticker} (need ≥60 rows)")
            return None

        X, _, _ = prepare_sequence_data(df_features, feature_cols, scaler=scaler)
        X_input = X[-1:]  # last sequence

        # News Sentiment Adjustment
        articles = fetch_news(ticker)
        sentiment_score = compute_sentiment_score(articles)
        sentiment_adjustment_factor = 0.02

        # Ensemble Prediction
        model_list = [(name, model) for name, model in models.items()]
        preds = []
        for name, model in model_list:
            pred = model.predict(X_input)
            preds.append(pred[0, 0] if pred.shape[-1] == 1 else pred[0])
        ensemble_pred = np.mean(preds)

        # Sentiment-adjusted prediction
        ensemble_pred = ensemble_pred * (1 + sentiment_score * sentiment_adjustment_factor)

        # Inverse transform
        scaled_pred = np.zeros((1, len(feature_cols)))
        scaled_pred[0, :] = X_input[0, -1, :]
        scaled_pred[0, feature_cols.index("Close")] = ensemble_pred
        inv_pred = scaler.inverse_transform(scaled_pred)[0, feature_cols.index("Close")]

        current_price = df['Close'].iloc[-1]
        delta = inv_pred - current_price
        confidence = abs(delta) / current_price
        direction = "Buy" if delta > 0 else "Sell"

        # Optional Multi-step Forecast
        forecast_prices = None
        if multi_steps > 0:
            forecast_prices = multi_step_forecast(model_list, X_input, scaler, feature_cols, steps=multi_steps)

        # Calculate Stop Loss and Take Profit
        if direction == "Buy":
            sl = current_price * (1 - sl_percent / 100)
            tp = current_price * (1 + tp_percent / 100)
        else:
            sl = current_price * (1 + sl_percent / 100)
            tp = current_price * (1 - tp_percent / 100)

        return {
            "current_price": current_price,
            "predicted_price": inv_pred,
            "confidence": confidence,
            "direction": direction,
            "stop_loss": sl,
            "take_profit": tp,
            "forecast_prices": forecast_prices,
            "sentiment_score": sentiment_score
        }

    except Exception as e:
        st.error(f"❌ Error in prediction: {e}")
        return None

def multi_step_forecast(models, X_input, scaler, feature_cols, steps=10):
    """Perform recursive multi-step forecasting."""
    forecasts = []
    X_current = X_input.copy()

    for _ in range(steps):
        preds = []
        for _, model in models:
            pred = model.predict(X_current)
            preds.append(pred[0, 0] if pred.shape[-1] == 1 else pred[0])
        ensemble_pred = np.mean(preds)

        # Inverse transform
        scaled_pred = np.zeros((1, len(feature_cols)))
        scaled_pred[0, :] = X_current[0, -1, :]
        scaled_pred[0, feature_cols.index("Close")] = ensemble_pred
        inv_pred = scaler.inverse_transform(scaled_pred)[0, feature_cols.index("Close")]

        forecasts.append(inv_pred)

        # Update input for next step
        next_features = X_current[0, 1:, :].copy()  # Shift left
        new_step = X_current[0, -1:, :].copy()      # Copy last step
        new_step[0, feature_cols.index("Close")] = ensemble_pred  # Update predicted close
        X_current = np.concatenate([next_features, new_step], axis=0)
        X_current = X_current.reshape(1, X_current.shape[0], X_current.shape[1])

    return forecasts

def plot_forecast_prices(df, forecast_prices):
    """Plot actual price and future forecasted prices."""
    fig = go.Figure()

    # Actual historical Close price
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Close'],
        mode='lines',
        name='Actual Price',
        line=dict(color='blue')
    ))

    # Forecasted future prices
    future_steps = len(forecast_prices)
    future_times = pd.date_range(
        start=df.index[-1] + pd.Timedelta(minutes=15),
        periods=future_steps,
        freq='15min'
    )

    fig.add_trace(go.Scatter(
        x=future_times,
        y=forecast_prices,
        mode='lines+markers',
        name='Forecasted Price',
        line=dict(color='orange', dash='dash')
    ))

    fig.update_layout(
        title="Price Forecast",
        xaxis_title="Time",
        yaxis_title="Price",
        template="plotly_white",
        showlegend=True
    )
    st.plotly_chart(fig)

# --- Signal Classification ---
def classify_signal(confidence, direction, sentiment_score):
    """Smarter signal classification including sentiment."""
    if abs(sentiment_score) > 0.3:
        sentiment_strength = "🚀 Very Strong"
    elif abs(sentiment_score) > 0.1:
        sentiment_strength = "⚡ Strong"
    else:
        sentiment_strength = "➖ Neutral"

    if confidence > 0.03:
        return f"{sentiment_strength} {direction.upper()}"
    elif confidence > 0.015:
        return f"Moderate {direction.upper()}"
    else:
        return f"Weak {direction.upper()}"

# --- Display Functions ---
def display_prediction_tables(ticker, pred, df=None):
    """Display prediction results, trading setup, and forecast if available."""
    if pred is None:
        st.warning(f"⚠️ No prediction available for {ticker}")
        return

    sl = pred['stop_loss']
    tp = pred['take_profit']
    signal_strength = classify_signal(pred['confidence'], pred['direction'], pred['sentiment_score'])

    # Metrics Table
    metrics = pd.DataFrame([
        {"Metric": "Current Price", "Value": f"${pred['current_price']:.2f}"},
        {"Metric": "Ensemble Predicted Price", "Value": f"${pred['predicted_price']:.2f}"},
        {"Metric": "Signal", "Value": signal_strength},
        {"Metric": "Confidence", "Value": f"{pred['confidence']*100:.2f}%"},
        {"Metric": "Sentiment Score", "Value": f"{pred['sentiment_score']:.2f}"},
    ])

    # Trade Setup Table
    risk_reward_ratio = abs(tp - pred['current_price']) / abs(pred['current_price'] - sl) if (pred['current_price'] - sl) != 0 else "N/A"
    setup = pd.DataFrame([
        {"Setup": "Stop Loss", "Level": f"${sl:.2f}"},
        {"Setup": "Take Profit", "Level": f"${tp:.2f}"},
        {"Setup": "Risk/Reward", "Level": f"{risk_reward_ratio:.2f}" if isinstance(risk_reward_ratio, float) else risk_reward_ratio},
    ])

    potential_profit = abs(tp - pred['current_price'])

    st.markdown(f"### 📌 Prediction for `{ticker}`")
    st.table(metrics)
    st.markdown("### 🛠️ Trade Setup")
    st.table(setup)
    st.markdown("### 💰 Potential Profit")
    st.write(f"Potential Profit: ${potential_profit:.2f}")
    st.markdown("---")

    # Show Forecast if available
    if pred.get('forecast_prices') is not None and df is not None:
        st.markdown("### 🔮 Multi-step Forecasted Prices")
        forecast_steps = pd.date_range(
            start=df.index[-1] + pd.Timedelta(minutes=15),
            periods=len(pred['forecast_prices']),
            freq='15min'
        )
        forecast_df = pd.DataFrame({
            'Time': forecast_steps,
            'Forecasted Price': pred['forecast_prices']
        })
        st.dataframe(forecast_df.style.format({"Forecasted Price": "{:.2f}"}))
        
        # Plot the forecast too
        plot_forecast_prices(df, pred['forecast_prices'])

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Settings")

market_type = st.sidebar.selectbox("Market Type", ["Stocks", "Forex", "Commodities"])
trained_tickers = {
    "Stocks": ["^GDAXI"],
    "Forex": ["USDJPY", "CHFJPY"],
    "Commodities": ["GC=F", "SI=F", "CL=F", "CC=F", "NG=F", "KC=F", "HG=F"]
}

tickers = st.sidebar.multiselect(
    "Select Tickers",
    trained_tickers[market_type],
    default=trained_tickers[market_type][:1]
)

sl_percent = st.sidebar.slider("Stop Loss %", 0.5, 5.0, 1.0)
tp_percent = st.sidebar.slider("Take Profit %", 1.0, 10.0, 2.0)

# --- Main Execution ---
if tickers:
    for ticker in tickers:
        df = fetch_historical_data(ticker)
        if df.empty:
            st.warning(f"⚠️ No data available for {ticker}")
            continue

        pred = predict_price(df, ticker, sl_percent, tp_percent, multi_steps=20)
        display_prediction_tables(ticker, pred, df)


else:
    st.info("👈 Please select at least one ticker from the sidebar.")
