import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import joblib
from collect_data import calculate_technical_indicators

def get_latest_data():
    """Get the latest Bitcoin data and calculate technical indicators."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=60)  # Get 60 days of data for indicators
    
    btc = yf.download('BTC-USD', start=start_date, end=end_date, interval='1d')
    df = calculate_technical_indicators(btc)
    return df.dropna()

def make_prediction():
    """Make a prediction for the next day's Bitcoin price."""
    try:
        # Load the model and scaler
        model = joblib.load('models/bitcoin_predictor.joblib')
        scaler = joblib.load('models/scaler.joblib')
        
        # Get latest data
        df = get_latest_data()
        
        # Prepare features
        feature_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'Returns', 'MA7', 'MA30', 'RSI', 'MACD',
            'Signal_Line', 'BB_Middle', 'BB_Upper', 'BB_Lower',
            'Volatility'
        ]
        
        # Get the latest data point
        latest_data = df[feature_columns].iloc[-1:].copy()
        
        # Scale the features
        latest_data_scaled = scaler.transform(latest_data)
        
        # Make prediction
        prediction = model.predict(latest_data_scaled)[0]
        
        # Get current price
        current_price = df['Close'].iloc[-1]
        
        # Calculate predicted change
        price_change = prediction - current_price
        percent_change = (price_change / current_price) * 100
        
        print("\nBitcoin Price Prediction")
        print("-" * 30)
        print(f"Current Price: ${current_price:.2f}")
        print(f"Predicted Price: ${prediction:.2f}")
        print(f"Predicted Change: ${price_change:.2f}")
        print(f"Predicted Change %: {percent_change:.2f}%")
        
        return prediction
        
    except FileNotFoundError:
        print("Error: Model files not found. Please run train_model.py first.")
    except Exception as e:
        print(f"Error making prediction: {str(e)}")

if __name__ == "__main__":
    make_prediction() 