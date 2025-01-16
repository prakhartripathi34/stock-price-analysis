import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

def prepare_lstm_data(df, feature_cols, target_col='close', window_size=60):
    """
    Prepares data for LSTM by creating sequences of length window_size.
    Returns train/test sets plus the scaler for inverse transformations.
    """
    df = df.dropna().copy()
    data = df[feature_cols + [target_col]].values

    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(window_size, len(data_scaled)):
        X.append(data_scaled[i - window_size:i, :-1])  # all features except last col (the target)
        y.append(data_scaled[i, -1])                  # last column is the target

    X, y = np.array(X), np.array(y)
    
    # Simple train/test split
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    return X_train, y_train, X_test, y_test, scaler

def build_lstm_model(input_shape):
    """
    Build a simple LSTM model for time-series forecasting.
    input_shape should be (window_size, num_features).
    """
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(64))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

if __name__ == '__main__':
    from data_collection import fetch_stock_data
    from feature_engineering import calculate_technical_indicators, add_dummy_sentiment
    
    # Fetch data
    symbol = 'AAPL'
    df = fetch_stock_data(symbol)
    df = calculate_technical_indicators(df)
    df = add_dummy_sentiment(df)
    
    # Prepare LSTM data
    feature_cols = ['ma_5', 'ma_20', 'rsi', 'sentiment']
    X_train, y_train, X_test, y_test, scaler = prepare_lstm_data(df, feature_cols, 'close', window_size=60)
    
    # Build, train, and evaluate the LSTM
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, epochs=1, batch_size=32, validation_split=0.1)  # Short training for demo
    
    loss = model.evaluate(X_test, y_test)
    print("Test Loss:", loss)
