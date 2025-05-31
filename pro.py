# pro.py

import os
import joblib
import requests
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
import torch

from model import (
    CNNLSTMAttention,
    TimeSeriesTransformer,
    TCN,
    SimpleInformer,
    enhance_features,
    prepare_sequence_data,
)

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

# --- Fetch News (Optional, can be skipped if not used) ---
def fetch_news(symbol):
    """Fetch news articles related to the given symbol."""
    key = os.getenv("NEWS_API_KEY")
    if not key:
        return []
    url = f'https://newsapi.org/v2/everything?q={symbol}&apiKey={key}'
    try:
        r = requests.get(url)
        r.raise_for_status()
        data = r.json()
        articles = [article['description'] for article in data.get('articles', [])]
        return articles
    except Exception:
        return []

# --- Compute Sentiment Score (Optional) ---
def compute_sentiment_score(articles):
    try:
        from nltk.sentiment import SentimentIntensityAnalyzer
        import nltk
        nltk.download('vader_lexicon', quiet=True)
        sia = SentimentIntensityAnalyzer()
        scores = [sia.polarity_scores(article)['compound'] for article in articles if article]
        return np.mean(scores) if scores else 0.0
    except Exception:
        return 0.0

# --- Load Models Function ---
def load_pytorch_models(ticker, feature_cols, device="cpu"):
    model_dir = Path("model")
    models = {}
    model_defs = {
        "cnn_lstm": CNNLSTMAttention(n_features=len(feature_cols)),
        "transformer": TimeSeriesTransformer(n_features=len(feature_cols)),
        "tcn": TCN(num_inputs=len(feature_cols)),
        "informer": SimpleInformer(input_size=len(feature_cols)),
    }
    for name, model in model_defs.items():
        model_path = model_dir / f"{ticker}_{name}.pt"
        if not model_path.exists():
            continue
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        models[name] = model.to(device)
    return models

# --- Prediction Function ---
def predict_price(df, ticker, sl_percent, tp_percent, multi_steps=0, device="cpu"):
    scaler_path = Path("model") / f"{ticker}_scaler.pkl"
    features_path = Path("model") / f"{ticker}_features.pkl"
    if not scaler_path.exists() or not features_path.exists():
        st.error(f"Scaler or features file not found for {ticker}")
        return None

    scaler = joblib.load(scaler_path)
    feature_cols = joblib.load(features_path)
    models = load_pytorch_models(ticker, feature_cols, device=device)
    if not models:
        st.error(f"No models loaded for {ticker}")
        return None

    try:
        df_features = enhance_features(df, feature_cols)
        if df_features is None or len(df_features) < 60:
            st.warning(f"⚠️ Not enough data for {ticker} (need ≥60 rows)")
            return None

        X, _, _ = prepare_sequence_data(df_features, feature_cols, scaler=scaler)
        X_input = X[-1:]  # last sequence
        X_tensor = torch.tensor(X_input, dtype=torch.float32).to(device)

        # Ensemble Prediction
        preds = []
        for model in models.values():
            with torch.no_grad():
                pred = model(X_tensor).cpu().numpy()
                pred_scalar = float(np.ravel(pred)[0])
                preds.append(pred_scalar)
        ensemble_pred = np.mean(preds)

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
            forecast_prices = multi_step_forecast(models, X_input, scaler, feature_cols, steps=multi_steps, device=device)

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
            "sentiment_score": 0.0,  # Add this line
        }

    except Exception as e:
        st.error(f"❌ Error in prediction: {e}")
        import traceback
        st.text(traceback.format_exc())
        return None

def multi_step_forecast(models, X_input, scaler, feature_cols, steps=10, device="cpu"):
    forecasts = []
    X_current = X_input.copy()
    for _ in range(steps):
        preds = []
        X_tensor = torch.tensor(X_current, dtype=torch.float32).to(device)
        for model in models.values():
            with torch.no_grad():
                pred = model(X_tensor).cpu().numpy()
                pred_scalar = float(np.ravel(pred)[0])
                preds.append(pred_scalar)
        ensemble_pred = np.mean(preds)
        # Inverse transform
        scaled_pred = np.zeros((1, len(feature_cols)))
        scaled_pred[0, :] = X_current[0, -1, :]
        scaled_pred[0, feature_cols.index("Close")] = ensemble_pred
        inv_pred = scaler.inverse_transform(scaled_pred)[0, feature_cols.index("Close")]
        forecasts.append(inv_pred)
        # Update input for next step
        next_features = X_current[0, 1:, :].copy()
        new_step = X_current[0, -1:, :].copy()
        new_step[0, feature_cols.index("Close")] = ensemble_pred
        X_current = np.concatenate([next_features, new_step], axis=0)
        X_current = X_current.reshape(1, X_current.shape[0], X_current.shape[1])
    return forecasts

def plot_forecast_prices(df, forecast_prices):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Close'],
        mode='lines',
        name='Actual Price',
        line=dict(color='blue')
    ))
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
# pro.py

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
    "Stocks": [],
    "Forex": [],
    "Commodities": ["CC=F"]
}

tickers = st.sidebar.multiselect(
    "Select Tickers",
    trained_tickers[market_type],
    default=trained_tickers[market_type][:1]
)

sl_percent = st.sidebar.slider("Stop Loss %", 0.5, 5.0, 1.0)
tp_percent = st.sidebar.slider("Take Profit %", 1.0, 10.0, 2.0)

st.sidebar.markdown("""
---
### ⚠️ Disclaimer

This application is provided for **educational and informational purposes only** and does not constitute financial, investment, or trading advice.  
Any decisions based on the output of this tool are **made at your own risk**.  
The developer is **not liable** for any losses or damages resulting from use of this application.

---
""")

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