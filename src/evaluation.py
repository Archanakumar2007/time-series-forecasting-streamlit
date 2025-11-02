from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def evaluate_models(actual, arima_forecast, sarima_forecast):
    """
    Evaluates ARIMA and SARIMA models using MAE and RMSE metrics.
    """
    print("\n📊 Model Evaluation Results:\n")

    # ARIMA evaluation
    arima_mae = mean_absolute_error(actual, arima_forecast)
    arima_rmse = np.sqrt(mean_squared_error(actual, arima_forecast))
    print(f"ARIMA -> MAE: {arima_mae:.2f}, RMSE: {arima_rmse:.2f}")

    # SARIMA evaluation
    sarima_mae = mean_absolute_error(actual, sarima_forecast)
    sarima_rmse = np.sqrt(mean_squared_error(actual, sarima_forecast))
    print(f"SARIMA -> MAE: {sarima_mae:.2f}, RMSE: {sarima_rmse:.2f}")

    # Return results for any further use
    return {
        "ARIMA": {"MAE": arima_mae, "RMSE": arima_rmse},
        "SARIMA": {"MAE": sarima_mae, "RMSE": sarima_rmse}
    }
