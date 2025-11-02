import matplotlib.pyplot as plt
import os

def plot_forecasts(train, test, arima_forecast, sarima_forecast, save_dir="results"):
    """Plots ARIMA, SARIMA, and combined forecasts with proper labels and saves to results/ folder."""

    # --- Plot ARIMA Forecast ---
    plt.figure(figsize=(10, 5))
    plt.plot(train.index, train['Sales'], label='Train', color='blue')
    plt.plot(test.index, test['Sales'], label='Test', color='black')
    plt.plot(test.index, arima_forecast, label='ARIMA Forecast', color='orange')
    plt.title("ARIMA Forecast")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "arima_forecast.png"))
    plt.close()

    # --- Plot SARIMA Forecast ---
    plt.figure(figsize=(10, 5))
    plt.plot(train.index, train['Sales'], label='Train', color='blue')
    plt.plot(test.index, test['Sales'], label='Test', color='black')
    plt.plot(test.index, sarima_forecast, label='SARIMA Forecast', color='green')
    plt.title("SARIMA Forecast")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "sarima_forecast.png"))
    plt.close()

    # --- Plot Combined Forecast ---
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train['Sales'], label='Train', color='blue')
    plt.plot(test.index, test['Sales'], label='Test', color='black')
    plt.plot(test.index, arima_forecast, label='ARIMA Forecast', color='orange')
    plt.plot(test.index, sarima_forecast, label='SARIMA Forecast', color='green')
    plt.title("ARIMA vs SARIMA Forecast Comparison")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "combined_forecast.png"))
    plt.show()
