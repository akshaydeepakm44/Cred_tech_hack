# app/dashboard.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import shap
import plotly.express as px

from app import config
from app.data_ingestion import get_stock_data, get_macro_data, get_news_sentiment
from app.model import prepare_features, load_model
from app.explain import explain_with_shap, explain_with_rules

# ========================================
# Streamlit Dashboard
# ========================================
st.set_page_config(page_title="Explainable Credit Intelligence", layout="wide")

st.title("💳 Explainable Credit Intelligence Platform")
st.write("An interpretable platform combining stock, macroeconomic, and news sentiment data to generate credit scores with explanations.")

# Sidebar inputs
st.sidebar.header("User Input")
ticker = st.sidebar.text_input("Enter Stock Ticker", config.DEFAULT_TICKER)
company = st.sidebar.text_input("Enter Company Name", config.DEFAULT_COMPANY)
macro_indicator = st.sidebar.selectbox("Macro Indicator", config.MACRO_INDICATORS)
model_path = st.sidebar.text_input("Model Path", f"{config.MODEL_DIR}/decision_tree_credit_model.pkl")
explain_mode = st.sidebar.radio("Explanation Mode", ["SHAP", "Rules"])

# Load trained model
try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"Model not found at {model_path}. Train a model first. Error: {e}")
    st.stop()

# Ingest data
with st.spinner("Fetching data..."):
    stock_df = get_stock_data(ticker, period=config.STOCK_PERIOD, interval=config.STOCK_INTERVAL)
    macro_df = get_macro_data(macro_indicator)
    news_df = get_news_sentiment(company)

# Prepare features
features = prepare_features(stock_df, macro_df, news_df)

if features.empty:
    st.warning("No features generated. Please try again with different inputs.")
    st.stop()

# Predict credit score
prediction = model.predict(features)[0]
score_label = "✅ Good Credit" if prediction == 1 else "❌ Bad Credit"

st.subheader(f"Credit Score Prediction for {company} ({ticker})")
st.metric("Predicted Creditworthiness", score_label)

# Explanations
st.subheader("Model Explanation")
if explain_mode == "SHAP":
    shap_importance = explain_with_shap(model, features)
    st.write("**Top Feature Contributions (SHAP values):**")
    st.dataframe(shap_importance)

    fig = px.bar(shap_importance, x="importance", y="feature", orientation="h", title="Feature Importance (SHAP)")
    st.plotly_chart(fig, use_container_width=True)

else:
    rules = explain_with_rules(model, features)
    st.write("**Decision Rules:**")
    for r in rules[:5]:
        st.write("-", r)

# Raw Data Tabs
st.subheader("📊 Data Sources")
tab1, tab2, tab3 = st.tabs(["Stock Data", "Macro Data", "News Sentiment"])

with tab1:
    st.dataframe(stock_df.tail())

with tab2:
    st.dataframe(macro_df.tail())

with tab3:
    st.dataframe(news_df.head())

