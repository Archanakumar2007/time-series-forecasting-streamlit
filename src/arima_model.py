import warnings
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")

def train_arima(train_series, order=(2,1,2), steps=8):
    """
    Trains an ARIMA model and forecasts future values.
    Args:
        train_series (pd.Series): training data (e.g., monthly sales)
        order (tuple): ARIMA(p,d,q)
        steps (int): number of periods to forecast
    Returns:
        forecast (np.ndarray): predicted values
        model (ARIMAResults): trained model object
    """
    print("📈 Training ARIMA model...")
    model = ARIMA(train_series, order=order)
    model_fit = model.fit()
    print("✅ ARIMA model trained successfully!")
    forecast = model_fit.forecast(steps=steps)
    return forecast, model_fit
