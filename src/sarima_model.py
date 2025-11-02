from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings("ignore")

def train_sarima(train_data, steps):
    """
    Train a SARIMA model and forecast future values.
    Parameters:
        train_data (pd.Series): Training time series data
        steps (int): Number of future periods to forecast
    Returns:
        forecast (pd.Series): SARIMA forecast values
        model_fit (SARIMAXResults): Fitted SARIMA model
    """
    print("🧠 Training SARIMA model...")

    # Configure the SARIMA model
    model = SARIMAX(
        train_data,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    # Fit the model
    model_fit = model.fit(disp=False)
    print("✅ SARIMA model trained successfully!")

    # Forecast future values
    forecast = model_fit.forecast(steps=steps)
    return forecast, model_fit
