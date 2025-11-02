import os
import sys
import pandas as pd
from datetime import datetime
from src.arima_model import train_arima
from src.sarima_model import train_sarima
from src.visualization import plot_forecasts
from src.data_loader import load_dataset
from src.evaluation import evaluate_models

# Ensure root folder is in system path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load dataset
data_path = os.path.join("data", "dataset.csv")
data = load_dataset(data_path)

# Split train-test
train_size = int(len(data) * 0.8)
train, test = data[:train_size], data[train_size:]

print("✅ Dataset Loaded Successfully!")
print(f"Training size: {len(train)}, Testing size: {len(test)}")

# Train ARIMA model
print("\n📈 Training ARIMA model...")
arima_forecast, arima_model = train_arima(train['Sales'], steps=len(test))
print("ARIMA model trained successfully!")

# Train SARIMA model
print("\n📊 Training SARIMA model...")
sarima_forecast, sarima_model = train_sarima(train['Sales'], steps=len(test))

print("SARIMA model trained successfully!")

# Evaluate Models
print("\n📉 Evaluating models...")
metrics = evaluate_models(test['Sales'], arima_forecast, sarima_forecast)
print(metrics)

# Plot Graphs
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

plot_forecasts(train, test, arima_forecast, sarima_forecast, save_dir=results_dir)

print("\n✅ Forecasting Completed! Check the 'results' folder for graphs and metrics.")
