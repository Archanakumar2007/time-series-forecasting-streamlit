# app.py
import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import os
import json

# import your existing functions from src
from src.data_loader import load_dataset
from src.arima_model import train_arima
from src.sarima_model import train_sarima
from src.evaluation import evaluate_models

st.set_page_config(page_title="Time Series Forecasting", layout="centered")

st.title("📈 Time Series Forecasting — ARIMA vs SARIMA")
st.write("Upload a CSV or use the dataset included in the repo. CSV must have `Date` and `Sales` columns.")

# --- helper to read dataset either from upload or repo ---
def load_dataframe(uploaded_file):
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            # try to normalize column names (your loader capitalizes)
            df.columns = [c.strip().capitalize() for c in df.columns]
            if 'Date' in df.columns and 'Sales' in df.columns:
                df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)
                df.set_index('Date', inplace=True)
                df = df.asfreq('MS', method='ffill')
                return df
            else:
                st.error("Uploaded CSV must contain 'Date' and 'Sales' columns (case-insensitive).")
                return None
        except Exception as e:
            st.error(f"Error reading uploaded file: {e}")
            return None
    else:
        # try loading included repo dataset
        default_path = os.path.join("data", "dataset.csv")
        if os.path.exists(default_path):
            try:
                return load_dataset(default_path)
            except Exception as e:
                st.error(f"Failed to load default dataset: {e}")
                return None
        else:
            st.info("No file uploaded and no local data/dataset.csv found in the repo.")
            return None

# --- UI: file uploader + controls ---
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
data = load_dataframe(uploaded_file)

if data is not None:
    st.subheader("Data Preview")
    st.dataframe(data.head())

    # train/test split control
    st.subheader("Split & Forecast Settings")
    train_pct = st.slider("Training set percentage", min_value=50, max_value=90, value=80)
    steps_override = st.number_input("Forecast horizon (steps). If 0, we use test length.", min_value=0, value=0)
    arima_p = st.number_input("ARIMA p (auto will be 2 if left)", min_value=0, max_value=5, value=2)
    arima_d = st.number_input("ARIMA d", min_value=0, max_value=2, value=1)
    arima_q = st.number_input("ARIMA q", min_value=0, max_value=5, value=2)

    run_button = st.button("Run Forecasts")

    if run_button:
        # split
        train_size = int(len(data) * train_pct / 100)
        train = data.iloc[:train_size].copy()
        test = data.iloc[train_size:].copy()
        if len(test) == 0:
            st.error("Test set is empty — reduce training percentage.")
        else:
            steps = steps_override if steps_override > 0 else len(test)
            st.info(f"Training length: {len(train)} — Testing length: {len(test)} — Forecasting {steps} steps")

            # --- Train ARIMA (use your train_arima function)
            try:
                arima_order = (int(arima_p), int(arima_d), int(arima_q))
                arima_forecast, arima_model = train_arima(train['Sales'], order=arima_order, steps=steps)
                # arima_forecast is typically a numpy array or pandas Series
                arima_forecast = np.array(arima_forecast).flatten()
            except Exception as e:
                st.error(f"ARIMA training failed: {e}")
                arima_forecast = None

            # --- Train SARIMA (use your train_sarima function)
            try:
                sarima_forecast, sarima_model = train_sarima(train['Sales'], steps=steps)
                sarima_forecast = np.array(sarima_forecast).flatten()
            except Exception as e:
                st.error(f"SARIMA training failed: {e}")
                sarima_forecast = None

            # --- Align forecasts to index for plotting
            forecast_index = pd.date_range(start=test.index[0], periods=steps, freq=test.index.inferred_freq or "MS")
            # if freq inference fails, fallback to monthly start
            if len(forecast_index) != steps:
                forecast_index = pd.date_range(start=test.index[0], periods=steps, freq='MS')

            if arima_forecast is not None and len(arima_forecast) == steps:
                arima_series = pd.Series(arima_forecast, index=forecast_index)
            else:
                arima_series = None

            if sarima_forecast is not None and len(sarima_forecast) == steps:
                sarima_series = pd.Series(sarima_forecast, index=forecast_index)
            else:
                sarima_series = None

            # --- Evaluate (use your evaluate_models if available)
            if arima_series is not None and sarima_series is not None:
                try:
                    metrics = evaluate_models(test['Sales'][:steps].values, arima_series.values, sarima_series.values)
                except Exception as e:
                    st.warning(f"Evaluation failed: {e}")
                    metrics = None
            else:
                metrics = None

            # --- Show metrics
            if metrics:
                st.subheader("Model Evaluation")
                st.write("MAE and RMSE for the two models:")
                st.json(metrics)

                # allow download of metrics
                buf = io.BytesIO()
                buf.write(json.dumps(metrics, indent=2).encode())
                buf.seek(0)
                st.download_button("Download metrics (JSON)", data=buf, file_name="metrics.json")

            # --- Plot results using matplotlib and show
            st.subheader("Forecast Comparison Plot")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(train.index, train['Sales'], label='Train', linewidth=1)
            ax.plot(test.index[:steps], test['Sales'][:steps], label='Test', linewidth=1, color='black')

            if arima_series is not None:
                ax.plot(arima_series.index, arima_series.values, label='ARIMA Forecast', linestyle='--')
            if sarima_series is not None:
                ax.plot(sarima_series.index, sarima_series.values, label='SARIMA Forecast', linestyle='--')

            ax.set_xlabel("Date")
            ax.set_ylabel("Sales")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

            # --- Optional: Show raw forecast numbers
            if arima_series is not None or sarima_series is not None:
                table = pd.DataFrame(index=forecast_index)
                if arima_series is not None:
                    table['ARIMA'] = arima_series.values
                if sarima_series is not None:
                    table['SARIMA'] = sarima_series.values
                st.subheader("Forecast values")
                st.dataframe(table)
else:
    st.info("Upload a CSV (or add data/dataset.csv in the repository) to get started.")
