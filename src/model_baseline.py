import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from pmdarima import arima

def linear_regression_predict(df, target_col='close', n_forward=1):
    """
    Train a simple Linear Regression model to predict n_forward days ahead price based on features.
    df should have columns: ['ma_5', 'ma_20', 'rsi', 'sentiment'] (or more).
    """
    df = df.dropna().copy()
    features = ['ma_5', 'ma_20', 'rsi', 'sentiment']
    
    if target_col not in df.columns:
        raise ValueError(f"'{target_col}' not found in DataFrame columns.")
    
    X = df[features]
    # The target is shifted n_forward days into the future
    y = df[target_col].shift(-n_forward)
    
    # Ensure we drop the last n_forward rows where y is NaN
    X = X[:-n_forward]
    y = y.dropna()
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Predictions: we predict only for the last n_forward rows in the original dataset
    preds = model.predict(df[features].tail(n_forward))
    
    return preds, model

def arima_predict(df, target_col='close', n_forward=1):
    """
    Fit an ARIMA model on the target_col and forecast n_forward steps.
    Uses pmdarima's auto_arima for convenience.
    """
    series = df[target_col].dropna()
    arima_model = arima.auto_arima(series, seasonal=False, trace=False)
    forecast = arima_model.predict(n_periods=n_forward)
    return forecast, arima_model

if __name__ == '__main__':
    from data_collection import fetch_stock_data
    from feature_engineering import calculate_technical_indicators, add_dummy_sentiment
    
    symbol = 'AAPL'
    stock_df = fetch_stock_data(symbol)
    stock_df = calculate_technical_indicators(stock_df)
    stock_df = add_dummy_sentiment(stock_df)

    # Linear Regression
    lr_preds, lr_model = linear_regression_predict(stock_df, n_forward=5)
    print("Linear Regression Predictions (5-day ahead):", lr_preds)
    
    # ARIMA
    arima_forecast, arima_model = arima_predict(stock_df, n_forward=5)
    print("ARIMA Forecast (5-day ahead):", arima_forecast)
