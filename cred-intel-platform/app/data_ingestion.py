# app/data_ingestion.py

import os
import pandas as pd
import yfinance as yf
import requests
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta

# Load config (API keys from environment variables or config.py later)
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "your_alpha_vantage_key")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "your_newsapi_key")


def get_stock_data(ticker: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    """
    Fetch historical stock price data using Yahoo Finance.
    """
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period, interval=interval)
    hist.reset_index(inplace=True)
    return hist


def get_macro_data(indicator: str = "GDP") -> pd.DataFrame:
    """
    Fetch macroeconomic data from Alpha Vantage.
    Options: 'GDP', 'CPI', 'INFLATION', 'UNEMPLOYMENT'
    """
    base_url = "https://www.alphavantage.co/query"
    indicator_map = {
        "GDP": "REAL_GDP",
        "CPI": "CPI",
        "INFLATION": "INFLATION",
        "UNEMPLOYMENT": "UNEMPLOYMENT"
    }

    if indicator not in indicator_map:
        raise ValueError(f"Indicator must be one of {list(indicator_map.keys())}")

    params = {
        "function": indicator_map[indicator],
        "apikey": ALPHA_VANTAGE_API_KEY
    }

    r = requests.get(base_url, params=params)
    data = r.json()

    # Convert JSON to DataFrame (structure depends on indicator)
    if "data" in data:
        df = pd.DataFrame(data["data"])
        df.rename(columns={"value": indicator}, inplace=True)
    elif "data" not in data and "data" in data.get("dataset", {}):
        df = pd.DataFrame(data["dataset"]["data"], columns=["date", indicator])
    else:
        # generic handler
        df = pd.DataFrame.from_dict(data, orient="index")

    return df


def get_news_sentiment(company_name: str, page_size: int = 10) -> pd.DataFrame:
    """
    Fetch latest news headlines about a company from NewsAPI
    and perform sentiment analysis using VADER.
    """
    url = "https://newsapi.org/v2/everything"
    today = datetime.now()
    from_date = (today - timedelta(days=7)).strftime("%Y-%m-%d")

    params = {
        "q": company_name,
        "from": from_date,
        "sortBy": "relevancy",
        "pageSize": page_size,
        "apiKey": NEWS_API_KEY,
        "language": "en"
    }

    r = requests.get(url, params=params)
    articles = r.json().get("articles", [])

    if not articles:
        return pd.DataFrame()

    analyzer = SentimentIntensityAnalyzer()

    rows = []
    for a in articles:
        title = a["title"]
        description = a.get("description", "")
        content = f"{title}. {description}"

        sentiment_score = analyzer.polarity_scores(content)["compound"]
        polarity = TextBlob(content).sentiment.polarity

        rows.append({
            "title": title,
            "publishedAt": a["publishedAt"],
            "source": a["source"]["name"],
            "url": a["url"],
            "vader_score": sentiment_score,
            "textblob_polarity": polarity
        })

    return pd.DataFrame(rows)


# Run standalone tests
if __name__ == "__main__":
    print("=== Stock Data ===")
    print(get_stock_data("AAPL").head())

    print("\n=== Macro Data (GDP) ===")
    try:
        print(get_macro_data("GDP").head())
    except Exception as e:
        print("Alpha Vantage Error:", e)

    print("\n=== News Sentiment (Apple) ===")
    print(get_news_sentiment("Apple").head())
