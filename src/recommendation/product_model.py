"""Product recommendation: predicts the product category a customer is likely to buy.

Train and evaluate:  python -m src.recommendation.product_model
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import MERGED_CSV, PRODUCT_MODEL

# Columns that identify a row or contain the answer, so the model must not use them
NOT_FEATURES = ["customer_id", "transaction_id", "purchase_date", "product_category"]


def make_models():
    return {
        "Baseline (always the most common category)": DummyClassifier(strategy="prior"),
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=42),
    }


def evaluate(model, X, y, customers, classes):
    """Test 5 times with different splits, keeping each customer's rows together, and average the scores."""
    scores = {"accuracy": [], "f1": [], "log_loss": []}
    confusion = np.zeros((len(classes), len(classes)), dtype=int)
    for seed in range(5):
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        probabilities = cross_val_predict(model, X, y, groups=customers, cv=splitter, method="predict_proba")
        predicted = classes[probabilities.argmax(axis=1)]
        scores["accuracy"].append(accuracy_score(y, predicted))
        scores["f1"].append(f1_score(y, predicted, average="weighted", zero_division=0))
        scores["log_loss"].append(log_loss(y, probabilities, labels=classes))
        confusion += confusion_matrix(y, predicted, labels=classes)

    result = {name: float(np.mean(values)) for name, values in scores.items()}
    result["accuracy_std"] = float(np.std(scores["accuracy"]))
    result["confusion_matrix"] = confusion.tolist()
    return result


def train():
    df = pd.read_csv(MERGED_CSV)
    X = df.drop(columns=NOT_FEATURES)
    y = df["product_category"]
    classes = np.array([str(c) for c in sorted(y.unique())])

    results = {name: evaluate(model, X, y, df["customer_id"], classes)
               for name, model in make_models().items()}

    # Pick the real model with the lowest log loss (the baseline is only there for comparison)
    candidates = [name for name in results if not name.startswith("Baseline")]
    best = min(candidates, key=lambda n: results[n]["log_loss"])

    model = make_models()[best].fit(X, y)
    PRODUCT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_columns": list(X.columns), "classes": classes.tolist()},
                PRODUCT_MODEL)

    report = {"selected_model": best, "n_rows": len(df), "n_customers": int(df["customer_id"].nunique()),
              "classes": classes.tolist(), "results": results}
    (PRODUCT_MODEL.parent / "metrics.json").write_text(json.dumps(report, indent=2))
    return report


def recommend(customer_id, top_n=3):
    """Return the top product categories for a customer, using their most recent purchase record."""
    bundle = joblib.load(PRODUCT_MODEL)
    df = pd.read_csv(MERGED_CSV)
    rows = df[df["customer_id"] == customer_id]
    if rows.empty:
        raise ValueError(f"Customer {customer_id} was not found in the dataset.")

    latest = rows.sort_values("purchase_date").iloc[[-1]]
    probabilities = bundle["model"].predict_proba(latest[bundle["feature_columns"]])[0]
    best = np.argsort(probabilities)[::-1][:top_n]
    return [(bundle["classes"][i], float(probabilities[i])) for i in best]


if __name__ == "__main__":
    report = train()
    print(f"Customers: {report['n_customers']} | rows: {report['n_rows']} | classes: {report['classes']}\n")
    print("Average of 5 customer-grouped test runs:")
    print(f"{'Model':45s} {'accuracy':>8s} {'+/-':>6s} {'F1':>7s} {'log loss':>9s}")
    for name, r in report["results"].items():
        print(f"{name:45s} {r['accuracy']:8.3f} {r['accuracy_std']:6.3f} {r['f1']:7.3f} {r['log_loss']:9.3f}")
    print("\nSelected model:", report["selected_model"])
