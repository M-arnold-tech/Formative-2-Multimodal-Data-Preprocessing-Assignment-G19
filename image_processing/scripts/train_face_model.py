import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
FEATURES_CSV = ROOT / "image_processing" / "data" / "image_features.csv"
MODEL_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "image_processing" / "models" / "face_model.pkl"
EXPRESSIONS = ["neutral", "smiling", "surprised"]
META_COLUMNS = ["member_id", "image_name", "expression", "augmentation", "face_found"]


def make_models():
    return {
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
    }


def evaluate(name, df, feature_cols):
    rows = []
    for held_out in EXPRESSIONS:
        train = df[df["expression"] != held_out]
        test = df[(df["expression"] == held_out) & (df["augmentation"] == "original")]
        model = make_models()[name].fit(train[feature_cols], train["member_id"])
        proba = model.predict_proba(test[feature_cols])
        pred = model.classes_[proba.argmax(axis=1)]
        rows.append({
            "model": name,
            "accuracy": accuracy_score(test["member_id"], pred),
            "f1": f1_score(test["member_id"], pred, average="weighted", zero_division=0),
            "log_loss": log_loss(test["member_id"], proba, labels=model.classes_),
        })
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(FEATURES_CSV)
    feature_cols = [c for c in df.columns if c not in META_COLUMNS]

    summary = pd.concat([evaluate(name, df, feature_cols) for name in make_models()])
    summary = summary.groupby("model")[["accuracy", "f1", "log_loss"]].mean().round(3)
    print(summary)
    best_name = summary.sort_values(["f1", "log_loss"], ascending=[False, True]).index[0]

    model = make_models()[best_name].fit(df[feature_cols], df["member_id"])
    gate_scaler = StandardScaler().fit(df[feature_cols])
    bundle = {
        "model": model,
        "model_name": best_name,
        "feature_cols": feature_cols,
        "threshold": 0.5,
        "classes": [str(c) for c in model.classes_],
        "gate_scaler": gate_scaler,
        "gate_matrix": gate_scaler.transform(df[feature_cols]),
    }

    if MODEL_PATH.exists():
        previous = joblib.load(MODEL_PATH)
        bundle["threshold"] = previous["threshold"]
        if "max_distance" in previous:
            bundle["max_distance"] = previous["max_distance"]

    joblib.dump(bundle, MODEL_PATH)
    print(f"Saved {best_name} to {MODEL_PATH}")


if __name__ == "__main__":
    main()
