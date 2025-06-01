import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf
import joblib
from collect_data import calculate_technical_indicators
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import plotly.express as px

# Set page config
st.set_page_config(
    page_title="Bitcoin Price Predictor",
    page_icon="₿",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .metric-box {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        margin: 5px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("₿ Bitcoin Price Predictor")
st.markdown("""
This app uses machine learning to predict Bitcoin prices based on historical data and technical indicators.
The model takes into account various factors including price movements, trading volume, and technical indicators.
""")

# Sidebar
st.sidebar.header("About")
st.sidebar.info("""
This is a machine learning model for educational purposes. 
Cryptocurrency prices are highly volatile and influenced by many factors. 
The predictions should not be used as financial advice.
""")

def get_latest_data():
    """Get the latest Bitcoin data and calculate technical indicators."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=60)  # Get 60 days of data for indicators
    
    btc = yf.download('BTC-USD', start=start_date, end=end_date, interval='1d')
    df = calculate_technical_indicators(btc)
    return df.dropna()

def plot_price_history(df):
    """Plot Bitcoin price history with technical indicators."""
    fig = go.Figure()
    
    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='BTC-USD'
    ))
    
    # Add Bollinger Bands
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['BB_Upper'],
        name='BB Upper',
        line=dict(color='rgba(250, 0, 0, 0.3)')
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['BB_Lower'],
        name='BB Lower',
        line=dict(color='rgba(0, 250, 0, 0.3)'),
        fill='tonexty'
    ))
    
    # Update layout
    fig.update_layout(
        title='Bitcoin Price History with Bollinger Bands',
        yaxis_title='Price (USD)',
        xaxis_title='Date',
        template='plotly_dark'
    )
    
    return fig

def evaluate_model(model, X_test, y_test, scaler):
    """Evaluate model performance and return metrics."""
    # Make predictions
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Calculate percentage error
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape,
        'predictions': y_pred,
        'actual': y_test
    }

def plot_predictions_vs_actual(eval_results):
    """Plot actual vs predicted values."""
    fig = px.scatter(
        x=eval_results['actual'],
        y=eval_results['predictions'],
        labels={'x': 'Actual Price', 'y': 'Predicted Price'},
        title='Actual vs Predicted Bitcoin Prices'
    )
    
    # Add perfect prediction line
    max_val = max(max(eval_results['actual']), max(eval_results['predictions']))
    min_val = min(min(eval_results['actual']), min(eval_results['predictions']))
    
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Perfect Prediction',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(template='plotly_dark')
    return fig

def main():
    try:
        # Load the model and scaler
        model = joblib.load('models/bitcoin_predictor.joblib')
        scaler = joblib.load('models/scaler.joblib')
        
        # Get latest data
        df = get_latest_data()
        
        # Display current price
        current_price = df['Close'].iloc[-1]
        st.metric("Current Bitcoin Price", f"${current_price:,.2f}")
        
        # Plot price history
        st.plotly_chart(plot_price_history(df), use_container_width=True)
        
        # Make prediction
        feature_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'Returns', 'MA7', 'MA30', 'RSI', 'MACD',
            'Signal_Line', 'BB_Middle', 'BB_Upper', 'BB_Lower',
            'Volatility'
        ]
        
        latest_data = df[feature_columns].iloc[-1:].copy()
        latest_data_scaled = scaler.transform(latest_data)
        prediction = model.predict(latest_data_scaled)[0]
        
        # Calculate predicted change
        price_change = prediction - current_price
        percent_change = (price_change / current_price) * 100
        
        # Display prediction
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Prediction for Tomorrow")
            st.markdown(f"""
            <div class="prediction-box">
                <h3>${prediction:,.2f}</h3>
                <p>Predicted Price</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### Expected Change")
            color = "green" if price_change >= 0 else "red"
            st.markdown(f"""
            <div class="prediction-box">
                <h3 style="color: {color}">${price_change:,.2f}</h3>
                <p>Price Change</p>
                <h3 style="color: {color}">{percent_change:.2f}%</h3>
                <p>Percentage Change</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Model Evaluation Section
        st.markdown("## Model Evaluation")
        
        # Load test data for evaluation
        test_data = pd.read_csv('data/bitcoin_data.csv', index_col=0, parse_dates=True)
        X_test = test_data[feature_columns].iloc[:-1]  # All but last row
        y_test = test_data['Close'].iloc[1:]  # All but first row
        
        # Evaluate model
        eval_results = evaluate_model(model, X_test, y_test, scaler)
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("### RMSE")
            st.markdown(f"""
            <div class="metric-box">
                <h3>${eval_results['RMSE']:,.2f}</h3>
                <p>Root Mean Square Error</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### MAE")
            st.markdown(f"""
            <div class="metric-box">
                <h3>${eval_results['MAE']:,.2f}</h3>
                <p>Mean Absolute Error</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("### R² Score")
            st.markdown(f"""
            <div class="metric-box">
                <h3>{eval_results['R2']:.4f}</h3>
                <p>Coefficient of Determination</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("### MAPE")
            st.markdown(f"""
            <div class="metric-box">
                <h3>{eval_results['MAPE']:.2f}%</h3>
                <p>Mean Absolute Percentage Error</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Plot predictions vs actual
        st.plotly_chart(plot_predictions_vs_actual(eval_results), use_container_width=True)
        
        # Display technical indicators
        st.markdown("### Technical Indicators")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("RSI", f"{df['RSI'].iloc[-1]:.2f}")
            st.metric("MACD", f"{df['MACD'].iloc[-1]:.2f}")
        
        with col2:
            st.metric("7-Day MA", f"${df['MA7'].iloc[-1]:,.2f}")
            st.metric("30-Day MA", f"${df['MA30'].iloc[-1]:,.2f}")
        
        with col3:
            st.metric("Volatility", f"{df['Volatility'].iloc[-1]:.4f}")
            st.metric("Signal Line", f"{df['Signal_Line'].iloc[-1]:.2f}")
        
    except FileNotFoundError:
        st.error("Model files not found. Please run train_model.py first.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 