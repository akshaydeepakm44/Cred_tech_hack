# Cred_tech_hack

🏦 Explainable Credit Intelligence Platform

1. Overview

Credit scoring is the backbone of lending decisions. Traditional models are often black boxes, hard to interpret, and rarely integrate real-time data like stock performance, macroeconomic conditions, or news sentiment.

This project introduces a data-driven, explainable credit intelligence system that is:

Transparent: Explains why a company is classified as high or low risk

Interpretable: Easy for stakeholders to understand

Data-rich: Uses multiple structured and unstructured data sources

2. Features

Fetches structured data: Stock prices (Yahoo Finance), GDP (Alpha Vantage)

Fetches unstructured data: Company news sentiment (NewsAPI)

Builds an interpretable Decision Tree model to classify credit risk

Uses Explainable AI (SHAP) for feature-level explanations

Interactive Streamlit dashboard for visualization

3. Technical Stack
Component	Technology	Why Chosen
Frontend Dashboard	Streamlit	Quick, hackathon-friendly UI
Data Sources	Yahoo Finance, Alpha Vantage, NewsAPI	Covers financial, macro, and sentiment data
ML Model	DecisionTreeClassifier	Interpretable, fast, explainable
Explainability	SHAP	Industry standard for feature importance
Sentiment Analysis	TextBlob + VADER	Easy-to-use, polarity and intensity analysis
Backend Logic	Python	Strong ecosystem for finance and ML
4. Installation & Run
git clone https://github.com/yourusername/credit-intelligence-platform.git
cd credit-intelligence-platform

# Add your API keys
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
NEWS_API_KEY=your_newsapi_key

# Run the Streamlit dashboard
streamlit run app.py

5. Code Flow

Imports & Config: Load all required libraries and store API keys

Data Ingestion:

get_stock_data: Downloads last 6 months of stock prices

get_macro_data: Fetches GDP data via Alpha Vantage

get_news_sentiment: Collects company news and computes sentiment scores

Model Training:

Creates features (returns, volatility)

Labels companies as High Risk (1) or Low Risk (0)

Trains a Decision Tree classifier

Explainability:

Uses SHAP TreeExplainer to calculate feature contributions

Visualizes which features influence credit risk predictions

Streamlit Dashboard:

Inputs: Stock ticker & company name

Visual outputs: Stock charts, GDP data, news sentiment, credit risk prediction, SHAP plots

6. Demo Flow

Enter a company ticker (e.g., AAPL)

System fetches:

Stock data

Macro indicators

News sentiment

Model predicts credit risk (Safe / Risky)

Dashboard shows why the prediction was made (e.g., high volatility → risky)

Decision-makers can trust and verify the model
