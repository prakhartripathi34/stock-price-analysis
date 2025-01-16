import streamlit as st
import pandas as pd
import numpy as np

from src.data_collection import fetch_stock_data
from src.feature_engineering import calculate_technical_indicators, add_dummy_sentiment
from src.model_baseline import linear_regression_predict, arima_predict
from src.model_lstm import prepare_lstm_data, build_lstm_model
from src.portfolio_optimization import mean_variance_optimization

@st.cache_data
def get_processed_data(symbol):
    df = fetch_stock_data(symbol=symbol)
    df = calculate_technical_indicators(df)
    df = add_dummy_sentiment(df)
    return df

def run_app():
    st.title("Stock Market Analysis and Portfolio Optimization")

    # User picks symbol and model
    symbol = st.text_input("Enter Stock Symbol:", "AAPL")
    model_choice = st.selectbox("Select Model:", ["Linear Regression", "ARIMA", "LSTM"])
    n_forward = st.number_input("Days Ahead to Predict:", min_value=1, max_value=30, value=5)

    if st.button("Run Analysis"):
        df = get_processed_data(symbol)
        st.subheader(f"Last 5 rows of data for {symbol}:")
        st.write(df.tail())

        if model_choice == "Linear Regression":
            preds, _ = linear_regression_predict(df, target_col='close', n_forward=n_forward)
            st.write(f"Linear Regression Predictions (Last {n_forward} days):")
            st.write(preds)

        elif model_choice == "ARIMA":
            forecast, _ = arima_predict(df, target_col='close', n_forward=n_forward)
            st.write(f"ARIMA Forecast (Next {n_forward} days):")
            st.write(forecast)

        else:  # LSTM
            feature_cols = ['ma_5', 'ma_20', 'rsi', 'sentiment']
            window_size = 60
            # Prepare data
            X_train, y_train, X_test, y_test, scaler = prepare_lstm_data(
                df, feature_cols, target_col='close', window_size=window_size
            )
            # Build model
            model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
            model.fit(X_train, y_train, epochs=1, batch_size=32, verbose=0)  # train quickly for demo
            # Predict the last n_forward steps (naive approach)
            # In practice, you might want a more robust forecast method
            if len(X_test) < n_forward:
                st.error("Not enough test data for the requested horizon.")
            else:
                predictions = model.predict(X_test[-n_forward:])
                st.write(f"LSTM Predictions for the last {n_forward} time windows:")
                st.write(predictions.flatten())

    # Portfolio optimization demo
    st.subheader("Portfolio Optimization")
    st.write("Example with dummy data for demonstration.")
    np.random.seed(42)
    dummy_returns = pd.DataFrame(np.random.randn(100, 4) / 100,
                                 columns=['AAPL', 'MSFT', 'GOOG', 'TSLA'])
    weights = mean_variance_optimization(dummy_returns)
    st.write("Optimized Portfolio Weights (Min Variance):")
    for stock, w in zip(dummy_returns.columns, weights):
        st.write(f"{stock}: {w:.4f}")

if __name__ == '__main__':
    run_app()
