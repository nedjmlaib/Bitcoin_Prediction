import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def flatten_columns(df):
    """Flatten multi-level column names."""
    df.columns = [col[0] for col in df.columns]
    return df

def calculate_technical_indicators(df):
    """Calculate technical indicators for the dataset."""
    # Flatten column names
    df = flatten_columns(df)
    
    # Print the columns we have
    print("Available columns:", df.columns.tolist())
    
    # Calculate returns
    df['Returns'] = df['Close'].pct_change()
    
    # Calculate moving averages
    df['MA7'] = df['Close'].rolling(window=7).mean()
    df['MA30'] = df['Close'].rolling(window=30).mean()
    
    # Calculate RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Calculate MACD (Moving Average Convergence Divergence)
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Calculate Bollinger Bands
    rolling_mean = df['Close'].rolling(window=20).mean()
    rolling_std = df['Close'].rolling(window=20).std()
    df['BB_Middle'] = rolling_mean
    df['BB_Upper'] = rolling_mean + (2 * rolling_std)
    df['BB_Lower'] = rolling_mean - (2 * rolling_std)
    
    # Calculate volatility
    df['Volatility'] = df['Returns'].rolling(window=20).std()
    
    return df

def collect_bitcoin_data():
    """Collect Bitcoin historical data and calculate technical indicators."""
    # Create data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # Download Bitcoin data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*2)  # 2 years of data
    
    print("Downloading Bitcoin data...")
    btc = yf.download('BTC-USD', start=start_date, end=end_date, interval='1d')
    
    # Print the shape and columns of the downloaded data
    print("\nDownloaded data shape:", btc.shape)
    print("Downloaded data columns:", btc.columns.tolist())
    
    if btc.empty:
        raise ValueError("No data was downloaded. Please check your internet connection and try again.")
    
    # Calculate technical indicators
    print("\nCalculating technical indicators...")
    df = calculate_technical_indicators(btc)
    
    # Drop NaN values
    df = df.dropna()
    
    # Print final data shape
    print("\nFinal data shape:", df.shape)
    print("Final data columns:", df.columns.tolist())
    
    # Save the processed data
    df.to_csv('data/bitcoin_data.csv')
    print(f"\nData collected and saved to data/bitcoin_data.csv")
    
    return df

if __name__ == "__main__":
    collect_bitcoin_data() 