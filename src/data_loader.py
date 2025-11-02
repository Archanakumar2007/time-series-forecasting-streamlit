import pandas as pd
import os

def load_dataset(filepath="data.csv"):
    """
    Loads a dataset with 'Date' and 'Sales' columns.
    Automatically parses the Date column as datetime and sets it as the index.
    """

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"❌ File not found: {filepath}")

    data = pd.read_csv(filepath)

    # Normalize column names (case-insensitive)
    data.columns = [col.strip().capitalize() for col in data.columns]

    # Validate required columns
    if 'Date' not in data.columns or 'Sales' not in data.columns:
        raise ValueError("❌ CSV must contain 'Date' and 'Sales' columns")

    # Convert 'Date' column to datetime
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    data = data.asfreq('MS')  # Monthly start frequency (optional)
    data = data.fillna(method='ffill')  # handle missing values
    return data
