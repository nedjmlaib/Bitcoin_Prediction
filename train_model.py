import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

def prepare_features(df):
    """Prepare features for the model."""
    # Select features for training
    feature_columns = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'Returns', 'MA7', 'MA30', 'RSI', 'MACD',
        'Signal_Line', 'BB_Middle', 'BB_Upper', 'BB_Lower',
        'Volatility'
    ]
    
    # Create target variable (next day's closing price)
    df['Target'] = df['Close'].shift(-1)
    
    # Drop the last row as we don't have the next day's price
    df = df[:-1]
    
    # Ensure all feature columns are numeric
    for col in feature_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop any rows with NaN values
    df = df.dropna()
    
    return df[feature_columns], df['Target']

def train_model():
    """Train the Random Forest model."""
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Load the data
    df = pd.read_csv('data/bitcoin_data.csv', index_col=0, parse_dates=True)
    
    # Remove any non-numeric columns
    df = df.select_dtypes(include=[np.number])
    
    # Prepare features and target
    X, y = prepare_features(df)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    train_predictions = model.predict(X_train_scaled)
    test_predictions = model.predict(X_test_scaled)
    
    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
    train_r2 = r2_score(y_train, train_predictions)
    test_r2 = r2_score(y_test, test_predictions)
    
    print(f"Training RMSE: ${train_rmse:.2f}")
    print(f"Testing RMSE: ${test_rmse:.2f}")
    print(f"Training R² Score: {train_r2:.4f}")
    print(f"Testing R² Score: {test_r2:.4f}")
    
    # Plot feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig('models/feature_importance.png')
    
    # Save the model and scaler
    joblib.dump(model, 'models/bitcoin_predictor.joblib')
    joblib.dump(scaler, 'models/scaler.joblib')
    
    print("\nModel and scaler saved to models/ directory")
    
    return model, scaler

if __name__ == "__main__":
    train_model() 