import pandas as pd
import numpy as np

def calculate_technical_indicators(df):
    """
    Calculate and append common technical indicators to the DataFrame.
    Assumes 'close' column exists in df.
    """
    df = df.copy()
    
    # Basic daily returns
    df['return'] = df['close'].pct_change()
    
    # Moving Averages
    df['ma_5'] = df['close'].rolling(window=5).mean()
    df['ma_20'] = df['close'].rolling(window=20).mean()
    
    # Bollinger Bands (using 20-day MA, 2 std)
    df['std_20'] = df['close'].rolling(window=20).std()
    df['upper_bb'] = df['ma_20'] + 2 * df['std_20']
    df['lower_bb'] = df['ma_20'] - 2 * df['std_20']
    
    # RSI Calculation (14-day)
    window_length = 14
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    
    ema_up = up.ewm(com=window_length - 1, adjust=False).mean()
    ema_down = down.ewm(com=window_length - 1, adjust=False).mean()
    rs = ema_up / ema_down
    df['rsi'] = 100 - (100 / (1 + rs))
    
    df.dropna(inplace=True)
    return df

def add_dummy_sentiment(df):
    """
    Adds a dummy sentiment column (0.0) for demonstration.
    If you have real sentiment data, replace this function with actual sentiment merging.
    """
    df['sentiment'] = 0.0
    return df

if __name__ == '__main__':
    from data_collection import fetch_stock_data
    df = fetch_stock_data(symbol='AAPL')
    df = calculate_technical_indicators(df)
    df = add_dummy_sentiment(df)
    print(df.head())
