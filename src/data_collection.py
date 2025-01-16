import os
import pandas as pd
import yfinance as yf

def fetch_stock_data(symbol='AAPL', start='2018-01-01', end='2023-01-01', 
                     data_dir='./data'):
    """
    Fetch historical stock data from Yahoo Finance for a given symbol.
    If a local CSV already exists, load from there instead of re-downloading.
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    csv_path = os.path.join(data_dir, f"{symbol}.csv")
    
    if os.path.exists(csv_path):
        # Load data from local CSV
        df = pd.read_csv(csv_path, parse_dates=True, index_col='Date')
    else:
        # Download data from Yahoo Finance
        df = yf.download(symbol, start=start, end=end)
        df.to_csv(csv_path)
    
    # Rename columns to lowercase for consistency
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
    # Ensure chronological order
    df.sort_index(inplace=True)
    
    return df

if __name__ == '__main__':
    # Quick test
    stock_df = fetch_stock_data(symbol='AAPL', start='2018-01-01', end='2023-01-01')
    print(stock_df.head())
