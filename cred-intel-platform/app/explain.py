# app/explain.py

import shap
import pandas as pd
import numpy as np
import joblib
from sklearn.tree import _tree
from app import config


def explain_with_shap(model, X: pd.DataFrame, max_display: int = 5):
    """
    Generate SHAP explanations for a given model and input features.
    Returns a DataFrame of feature importance values.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):  # for classifiers (multiple classes)
        shap_values = shap_values[1]  # focus on class 1 (good credit)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": X.columns,
        "importance": mean_abs_shap
    }).sort_values(by="importance", ascending=False)

    return importance_df.head(max_display)


def explain_with_rules(model, X: pd.DataFrame):
    """
    Extract simple human-readable rules from a DecisionTreeClassifier.
    Only works with DecisionTree, not RandomForest.
    """
    if not hasattr(model, "tree_"):
        raise ValueError("Rule-based explanations only work with DecisionTreeClassifier")

    tree_ = model.tree_
    feature_names = X.columns

    rules = []

    def recurse(node, depth, path):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_names[tree_.feature[node]]
            threshold = tree_.threshold[node]
            recurse(tree_.children_left[node], depth + 1,
                    path + [f"{name} <= {threshold:.2f}"])
            recurse(tree_.children_right[node], depth + 1,
                    path + [f"{name} > {threshold:.2f}"])
        else:
            value = tree_.value[node]
            class_label = np.argmax(value)
            rule_str = " AND ".join(path) if path else "ROOT"
            rules.append(f"IF {rule_str} THEN class = {class_label}")

    recurse(0, 1, [])
    return rules


def load_model_for_explain(model_path: str):
    """
    Load model and return for explainability.
    """
    return joblib.load(model_path)


# Standalone test
if __name__ == "__main__":
    from sklearn.tree import DecisionTreeClassifier

    # Dummy data
    X = pd.DataFrame({
        "avg_return": [0.02, -0.01, 0.03],
        "volatility": [0.05, 0.1, 0.08],
        "latest_close": [150, 145, 160],
        "avg_vader": [0.2, -0.1, 0.3],
        "avg_textblob": [0.1, -0.2, 0.15]
    })
    y = pd.Series([1, 0, 1])

    # Train simple decision tree
    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(X, y)

    # Test SHAP explanation
    shap_expl = explain_with_shap(model, X)
    print("=== SHAP Explanations ===")
    print(shap_expl)

    # Test Rule-based explanation
    rule_expl = explain_with_rules(model, X)
    print("\n=== Rule-based Rules ===")
    for r in rule_expl:
        print(r)
