"""Voice verification: checks that a recording belongs to the person the face check identified.

Train and evaluate:  python -m src.voice_verification.voice_model
"""
import json

import joblib
import librosa
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import AUDIO_DIR, VOICE_MODEL

# Every recording is resampled to this rate, so the phone or microphone used does not change the features
SAMPLE_RATE = 22050

# The audio files call this person "Keane", the face model calls them "gaju"
NAME_FIXES = {"keane": "gaju"}

# Same 30 features as the audio notebook: 13 MFCCs (mean and std), roll-off and RMS energy
FEATURE_COLUMNS = (
    [f"mfcc_{i}_{stat}" for i in range(1, 14) for stat in ("mean", "std")]
    + ["rolloff_mean", "rolloff_std", "rms_energy_mean", "rms_energy_std"]
)


def clean_name(name):
    name = name.strip().lower()
    return NAME_FIXES.get(name, name)


def features_from_signal(y):
    """Turn audio samples into the 30 numbers the model uses (same method as the notebook)."""
    mfccs = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=13)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=SAMPLE_RATE)
    rms = librosa.feature.rms(y=y)

    features = {}
    for i in range(13):
        features[f"mfcc_{i + 1}_mean"] = mfccs[i].mean()
        features[f"mfcc_{i + 1}_std"] = mfccs[i].std()
    features.update(rolloff_mean=rolloff.mean(), rolloff_std=rolloff.std(),
                    rms_energy_mean=rms.mean(), rms_energy_std=rms.std())
    return features


def extract_features(audio_path):
    y, _ = librosa.load(audio_path, sr=SAMPLE_RATE)
    return features_from_signal(y)


def augmented_versions(y):
    """The recording plus the three augmentations used in the audio notebook."""
    noise = np.random.default_rng(0).standard_normal(len(y))
    return {
        "none": y,
        "pitch_shift": librosa.effects.pitch_shift(y=y, sr=SAMPLE_RATE, n_steps=2),
        "time_stretch": librosa.effects.time_stretch(y=y, rate=1.15),
        "added_noise": y + 0.005 * noise,
    }


def split_file_name(file_name):
    """'Linda_yes_approve' -> ('Linda', 'yes_approve');  'Yes approve - Keane' -> ('Keane', 'yes_approve')."""
    if " - " in file_name:
        phrase, member = file_name.rsplit(" - ", 1)
    else:
        member, phrase = file_name.split("_", 1)
    return member, phrase.strip().lower().replace(" ", "_")


def build_training_table():
    """One row per version of each recording (8 recordings x 4 versions)."""
    recordings = sorted(AUDIO_DIR.glob("*.wav")) + sorted((AUDIO_DIR / "converted").glob("*.wav"))
    rows = []
    for path in recordings:
        member, phrase = split_file_name(path.stem)
        y, _ = librosa.load(path, sr=SAMPLE_RATE)
        for augmentation, signal in augmented_versions(y).items():
            rows.append({"member": clean_name(member), "phrase": phrase, "augmentation": augmentation,
                         **features_from_signal(signal)})
    return pd.DataFrame(rows)


def make_models():
    return {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
    }


def fit(name, rows):
    return make_models()[name].fit(rows[FEATURE_COLUMNS], rows["member"])


def evaluate(name, df, classes):
    """Train on one phrase and test on the other, so the model is tested on words it never heard."""
    true, predicted, probabilities = [], [], []
    for held_out in df["phrase"].unique():
        test = df[(df["phrase"] == held_out) & (df["augmentation"] == "none")]
        model = fit(name, df[df["phrase"] != held_out])
        proba = model.predict_proba(test[FEATURE_COLUMNS])
        true += list(test["member"])
        predicted += list(model.classes_[proba.argmax(axis=1)])
        probabilities += list(proba)
    return {
        "n_test_clips": len(true),
        "accuracy": accuracy_score(true, predicted),
        "f1": f1_score(true, predicted, average="weighted", zero_division=0),
        "log_loss": log_loss(true, probabilities, labels=classes),
        "confusion_matrix": confusion_matrix(true, predicted, labels=classes).tolist(),
    }


def stranger_confidence(name, df):
    """Hide one speaker at a time and see how confident the model is about that unknown voice."""
    confidences = []
    for member in df["member"].unique():
        test = df[(df["member"] == member) & (df["augmentation"] == "none")]
        model = fit(name, df[df["member"] != member])
        confidences += list(model.predict_proba(test[FEATURE_COLUMNS]).max(axis=1))
    return [round(float(c), 2) for c in confidences]


def train():
    df = build_training_table()
    classes = sorted(df["member"].unique())

    results = {name: evaluate(name, df, classes) for name in make_models()}
    best = min(results, key=lambda n: results[n]["log_loss"])   # lowest log loss wins

    VOICE_MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": fit(best, df), "feature_columns": FEATURE_COLUMNS}, VOICE_MODEL)

    report = {"selected_model": best, "members": classes, "results": results,
              "unknown_voice_confidence": stranger_confidence(best, df)}
    (VOICE_MODEL.parent / "metrics.json").write_text(json.dumps(report, indent=2))
    return report


def verify_voice(audio_path, claimed_member):
    """Return (is_verified, who_the_model_hears, confidence).

    Verified only when the voice is identified as the same person the face check found.
    """
    bundle = joblib.load(VOICE_MODEL)
    features = pd.DataFrame([extract_features(audio_path)])[bundle["feature_columns"]]
    probabilities = bundle["model"].predict_proba(features)[0]
    heard = str(bundle["model"].classes_[probabilities.argmax()])
    return heard == claimed_member, heard, float(probabilities.max())


if __name__ == "__main__":
    report = train()
    print("Members:", report["members"])
    print(f"{'Model':22s} {'clips':>5s} {'accuracy':>8s} {'F1':>7s} {'log loss':>9s}")
    for name, r in report["results"].items():
        print(f"{name:22s} {r['n_test_clips']:5d} {r['accuracy']:8.3f} {r['f1']:7.3f} {r['log_loss']:9.3f}")
    print("\nSelected model:", report["selected_model"])
    print("Confusion matrix (rows = true, columns = predicted):")
    for member, row in zip(report["members"], report["results"][report["selected_model"]]["confusion_matrix"]):
        print(f"  {member:8s} {row}")
    print("Confidence on voices the model has never heard:", report["unknown_voice_confidence"])
