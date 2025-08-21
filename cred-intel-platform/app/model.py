# app/model.py

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from app import config


def prepare_features(stock_df: pd.DataFrame, macro_df: pd.DataFrame, news_df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic feature engineering:
    - From stock: daily returns, volatility
    - From macro: latest values
    - From news: average sentiment
    Returns combined DataFrame (ready for model).
    """

    features = {}

    # --- Stock features ---
    if not stock_df.empty:
        stock_df["Return"] = stock_df["Close"].pct_change()
        features["avg_return"] = stock_df["Return"].mean()
        features["volatility"] = stock_df["Return"].std()
        features["latest_close"] = stock_df["Close"].iloc[-1]

    # --- Macro features ---
    if not macro_df.empty:
        macro_df = macro_df.rename(columns={macro_df.columns[-1]: "value"})
        macro_df["value"] = pd.to_numeric(macro_df["value"], errors="coerce")
        features[f"macro_{macro_df.columns[-1]}"] = macro_df["value"].iloc[-1]

    # --- News sentiment features ---
    if not news_df.empty:
        features["avg_vader"] = news_df["vader_score"].mean()
        features["avg_textblob"] = news_df["textblob_polarity"].mean()

    return pd.DataFrame([features])


def train_model(X: pd.DataFrame, y: pd.Series, model_type: str = "decision_tree"):
    """
    Train a simple interpretable model (Decision Tree or Random Forest).
    """
    if model_type == "decision_tree":
        model = DecisionTreeClassifier(max_depth=3, random_state=42)
    else:
        model = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Save model
    model_path = os.path.join(config.MODEL_DIR, f"{model_type}_credit_model.pkl")
    joblib.dump(model, model_path)
    print(f"Model saved at {model_path}")

    return model, model_path


def load_model(model_path: str):
    """
    Load a saved model.
    """
    return joblib.load(model_path)


# Example standalone run
if __name__ == "__main__":
    # Example dummy data (replace with real ingestion pipeline later)
    stock = pd.DataFrame({"Close": [150, 152, 151, 155, 160]})
    macro = pd.DataFrame({"date": ["2025-01-01", "2025-04-01"], "GDP": [2.3, 2.5]})
    news = pd.DataFrame({"vader_score": [0.2, -0.1, 0.3], "textblob_polarity": [0.1, -0.05, 0.2]})

    X = prepare_features(stock, macro, news)

    # Create dummy target (1 = Good credit, 0 = Bad credit)
    y = pd.Series([1])  # Hackathon demo — replace with real credit dataset

    # To train properly, you need more rows (loop ingestion across multiple tickers/companies)
    print("Prepared features:\n", X)
    print("Skipping training due to insufficient data in standalone demo.")
