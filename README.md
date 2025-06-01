# Bitcoin Price Predictor

This project implements a machine learning model to predict Bitcoin prices using historical data. The model uses various technical indicators and features to make predictions for the next day's Bitcoin price.

## Features
- Historical Bitcoin price data collection
- Feature engineering with technical indicators
- Machine learning model training
- Price prediction functionality
- Model evaluation and visualization

## Setup
1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the data collection script:
```bash
python collect_data.py
```

3. Train the model:
```bash
python train_model.py
```

4. Make predictions:
```bash
python predict.py
```

## Project Structure
- `collect_data.py`: Script to collect and preprocess Bitcoin price data
- `train_model.py`: Script to train the machine learning model
- `predict.py`: Script to make predictions using the trained model
- `requirements.txt`: Project dependencies
- `models/`: Directory containing saved model files
- `data/`: Directory containing collected and processed data

## Model Details
The project uses a Random Forest Regressor to predict Bitcoin prices. The model takes into account various technical indicators and historical price data to make predictions for the next day's price.
