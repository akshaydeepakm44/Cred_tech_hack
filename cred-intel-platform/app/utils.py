# app/utils.py

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder


# ==========================
# Data Cleaning & Preprocessing
# ==========================

def handle_missing(df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
    """
    Fill missing values in DataFrame using given strategy.
    Options: mean, median, mode, drop
    """
    df_copy = df.copy()
    if strategy == "mean":
        df_copy.fillna(df_copy.mean(), inplace=True)
    elif strategy == "median":
        df_copy.fillna(df_copy.median(), inplace=True)
    elif strategy == "mode":
        df_copy.fillna(df_copy.mode().iloc[0], inplace=True)
    elif strategy == "drop":
        df_copy.dropna(inplace=True)
    else:
        raise ValueError("Invalid strategy. Use 'mean', 'median', 'mode', or 'drop'")
    return df_copy


def scale_features(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Scale selected columns using StandardScaler.
    """
    scaler = StandardScaler()
    df[cols] = scaler.fit_transform(df[cols])
    return df


def encode_labels(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Label encode a categorical column.
    """
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col])
    return df


# ==========================
# Feature Engineering
# ==========================

def add_rolling_features(df: pd.DataFrame, col: str, windows=[3, 5, 10]) -> pd.DataFrame:
    """
    Add rolling mean and std for given column.
    """
    for w in windows:
        df[f"{col}_rollmean_{w}"] = df[col].rolling(window=w).mean()
        df[f"{col}_rollstd_{w}"] = df[col].rolling(window=w).std()
    return df


def calculate_volatility(df: pd.DataFrame, col: str = "Close") -> float:
    """
    Calculate volatility as standard deviation of returns.
    """
    if col not in df.columns:
        return np.nan
    df["Return"] = df[col].pct_change()
    return df["Return"].std()


# ==========================
# File Utilities
# ==========================

def save_df(df: pd.DataFrame, path: str):
    """
    Save DataFrame as CSV.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def load_df(path: str) -> pd.DataFrame:
    """
    Load DataFrame from CSV.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)


# ==========================
# Logging Utility
# ==========================

def log(msg: str):
    """
    Simple logger for console output.
    """
    print(f"[INFO] {msg}")


# Standalone test
if __name__ == "__main__":
    # Dummy DataFrame
    df = pd.DataFrame({
        "Close": [100, 102, np.nan, 105, 110],
        "Category": ["A", "B", "A", "C", "B"]
    })

    print("Original DF:\n", df)

    df = handle_missing(df, strategy="mean")
    df = encode_labels(df, "Category")
    df = add_rolling_features(df, "Close", windows=[2, 3])
    print("\nProcessed DF:\n", df)

    vol = calculate_volatility(df, "Close")
    print("\nVolatility:", vol)

    # Save & load
    save_df(df, "data/test_utils.csv")
    loaded = load_df("data/test_utils.csv")
    print("\nLoaded DF:\n", loaded.head())
