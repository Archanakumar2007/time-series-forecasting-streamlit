from http.server import BaseHTTPRequestHandler
import json
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        # Load dataset from data folder
        df = pd.read_csv("data/dataset.csv")
        
        # Fit a simple ARIMA model
        model = ARIMA(df["Sales"], order=(1, 1, 1))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=1)[0]

        # Return the forecast value
        response = {"next_month_forecast": float(forecast)}
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())
