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
st.write("Upload a CSV or use the default dataset. CSV must have `Date` and `Sales` columns.")

# ------------------------------
# Helper to load uploaded or default data
# ------------------------------
def load_dataframe(uploaded_file):
    try:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            # Normalize column names
            df.columns = [c.strip().capitalize() for c in df.columns]

            if 'Date' not in df.columns or 'Sales' not in df.columns:
                st.error("Uploaded CSV must contain 'Date' and 'Sales' columns (case-insensitive).")
                return None

            # Clean and convert
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True)
            df.dropna(subset=['Date'], inplace=True)

            # Handle duplicates safely
            df = df.groupby('Date', as_index=False, sort=True)['Sales'].sum()

            # Set index
            df.set_index('Date', inplace=True)

            # Ensure frequency is monthly (or infer)
            try:
                df = df.asfreq('MS')  # Monthly start
            except Exception:
                df = df.resample('MS').ffill()

            df = df.sort_index()
            return df

        else:
            # Load repo dataset if no file uploaded
            default_path = os.path.join("data", "dataset.csv")
            if os.path.exists(default_path):
                df = load_dataset(default_path)
                df = df.sort_index()
                return df
            else:
                st.info("No file uploaded and no local data/dataset.csv found.")
                return None

    except Exception as e:
        st.error(f"Error reading uploaded file: {e}")
        return None


# ------------------------------
# Streamlit UI
# ------------------------------
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
data = load_dataframe(uploaded_file)

if data is not None and not data.empty:
    st.subheader("Data Preview")
    st.dataframe(data.head())

    # ------------------------------
    # Model controls
    # ------------------------------
    st.subheader("Split & Forecast Settings")
    train_pct = st.slider("Training set percentage", min_value=50, max_value=90, value=80)
    steps_override = st.number_input("Forecast horizon (steps). If 0, uses test length.", min_value=0, value=0)
    arima_p = st.number_input("ARIMA p", min_value=0, max_value=5, value=2)
    arima_d = st.number_input("ARIMA d", min_value=0, max_value=2, value=1)
    arima_q = st.number_input("ARIMA q", min_value=0, max_value=5, value=2)

    if st.button("Run Forecasts"):
        # --- Train/test split ---
        train_size = int(len(data) * train_pct / 100)
        train = data.iloc[:train_size].copy()
        test = data.iloc[train_size:].copy()

        if len(test) == 0:
            st.error("Test set is empty — reduce training percentage.")
        else:
            steps = steps_override if steps_override > 0 else len(test)
            st.info(f"Training length: {len(train)} — Testing length: {len(test)} — Forecasting {steps} steps")

            # --- Train ARIMA ---
            try:
                arima_order = (int(arima_p), int(arima_d), int(arima_q))
                arima_forecast, _ = train_arima(train['Sales'], order=arima_order, steps=steps)
                arima_forecast = np.array(arima_forecast).flatten()
            except Exception as e:
                st.error(f"ARIMA training failed: {e}")
                arima_forecast = None

            # --- Train SARIMA ---
            try:
                sarima_forecast, _ = train_sarima(train['Sales'], steps=steps)
                sarima_forecast = np.array(sarima_forecast).flatten()
            except Exception as e:
                st.error(f"SARIMA training failed: {e}")
                sarima_forecast = None

            # --- Forecast index ---
            freq = test.index.inferred_freq or 'MS'
            forecast_index = pd.date_range(start=test.index[0], periods=steps, freq=freq)

            arima_series = pd.Series(arima_forecast, index=forecast_index) if arima_forecast is not None else None
            sarima_series = pd.Series(sarima_forecast, index=forecast_index) if sarima_forecast is not None else None

            # --- Evaluate ---
            metrics = None
            if arima_series is not None and sarima_series is not None:
                try:
                    metrics = evaluate_models(test['Sales'][:steps].values,
                                              arima_series.values,
                                              sarima_series.values)
                except Exception as e:
                    st.warning(f"Evaluation failed: {e}")

            # --- Display metrics ---
            if metrics:
                st.subheader("Model Evaluation")
                st.json(metrics)

                buf = io.BytesIO(json.dumps(metrics, indent=2).encode())
                st.download_button("Download metrics (JSON)", data=buf, file_name="metrics.json")

            # --- Plot ---
            st.subheader("Forecast Comparison Plot")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(train.index, train['Sales'], label='Train', linewidth=1)
            ax.plot(test.index[:steps], test['Sales'][:steps], label='Test', color='black', linewidth=1)
            if arima_series is not None:
                ax.plot(arima_series.index, arima_series.values, '--', label='ARIMA Forecast')
            if sarima_series is not None:
                ax.plot(sarima_series.index, sarima_series.values, '--', label='SARIMA Forecast')
            ax.set_xlabel("Date")
            ax.set_ylabel("Sales")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

            # --- Forecast Table ---
            st.subheader("Forecast Values")
            table = pd.DataFrame(index=forecast_index)
            if arima_series is not None:
                table['ARIMA'] = arima_series.values
            if sarima_series is not None:
                table['SARIMA'] = sarima_series.values
            st.dataframe(table)

else:
    st.info("Upload a CSV (or add data/dataset.csv) to start.")
