# app/config.py

import os

# ========================
# API KEYS (replace with your keys or set as ENV variables)
# ========================
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "your_alpha_vantage_key")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "your_newsapi_key")

# ========================
# Default settings
# ========================
DEFAULT_TICKER = "AAPL"
DEFAULT_COMPANY = "Apple"

# Time ranges for stock data
STOCK_PERIOD = "6mo"
STOCK_INTERVAL = "1d"

# Macro indicators available
MACRO_INDICATORS = ["GDP", "CPI", "INFLATION", "UNEMPLOYMENT"]

# ========================
# Paths
# ========================
DATA_DIR = os.path.join(os.getcwd(), "data")
MODEL_DIR = os.path.join(os.getcwd(), "models")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
