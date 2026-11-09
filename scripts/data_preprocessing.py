"""
data_preprocessing.py  --  Formative 2 (Person 1)
Merge customer_social_profiles + customer_transactions, clean, engineer features
and write data/merged_dataset.csv (target: product_category).

Usage:  python scripts/data_preprocessing.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SENTIMENT_MAP = {"Negative": -1, "Neutral": 0, "Positive": 1}
PLATFORMS = ["Facebook", "Instagram", "LinkedIn", "TikTok", "Twitter"]


def load_raw():
    social = pd.read_csv(DATA / "customer_social_profiles.csv")
    trans = pd.read_csv(DATA / "customer_transactions.csv")
    return social, trans


def clean_social(social: pd.DataFrame) -> pd.DataFrame:
    s = social.copy()
    # 1. Harmonise the key: 'A178' -> 178 so it matches customer_id_legacy (int)
    s["customer_id"] = s["customer_id_new"].str.extract(r"(\d+)").astype(int)
    s = s.drop(columns="customer_id_new")
    # 2. Standardise text, enforce types
    s["social_media_platform"] = s["social_media_platform"].str.strip().str.title().replace({"Linkedin": "LinkedIn", "Tiktok": "TikTok"})
    s["review_sentiment"] = s["review_sentiment"].str.strip().str.title()
    # 3. Drop exact duplicate rows (same customer/platform/scores/sentiment)
    s = s.drop_duplicates().reset_index(drop=True)
    # 4. Validate ranges
    assert s["engagement_score"].between(0, 100).all()
    assert s["purchase_interest_score"].between(0, 5).all()
    s["sentiment_score"] = s["review_sentiment"].map(SENTIMENT_MAP)
    return s


def aggregate_social(s: pd.DataFrame) -> pd.DataFrame:
    """Social table is many rows per customer -> collapse to ONE row per customer
    so that the join does not multiply transactions (avoids row explosion/leakage)."""
    g = s.groupby("customer_id")
    agg = g.agg(
        n_social_profiles=("social_media_platform", "size"),
        n_platforms=("social_media_platform", "nunique"),
        avg_engagement=("engagement_score", "mean"),
        max_engagement=("engagement_score", "max"),
        avg_purchase_interest=("purchase_interest_score", "mean"),
        max_purchase_interest=("purchase_interest_score", "max"),
        avg_sentiment=("sentiment_score", "mean"),
    )
    agg["top_platform"] = g["social_media_platform"].agg(lambda x: x.value_counts().sort_index().idxmax())
    # share of the customer's profiles on each platform
    share = pd.crosstab(s["customer_id"], s["social_media_platform"], normalize="index").reindex(columns=PLATFORMS, fill_value=0)
    share.columns = [f"platform_share_{c.lower()}" for c in share.columns]
    return agg.join(share).reset_index()


def clean_transactions(t: pd.DataFrame) -> pd.DataFrame:
    t = t.copy()
    t = t.rename(columns={"customer_id_legacy": "customer_id"})
    t["customer_id"] = t["customer_id"].astype(int)
    t["purchase_date"] = pd.to_datetime(t["purchase_date"], errors="raise")
    t["product_category"] = t["product_category"].str.strip().str.title()
    t = t.drop_duplicates(subset="transaction_id").reset_index(drop=True)
    assert (t["purchase_amount"] > 0).all()
    return t


def engineer(merged: pd.DataFrame) -> pd.DataFrame:
    m = merged.copy()
    # --- missing ratings: flag + impute with the global median (robust to skew)
    m["rating_missing"] = m["customer_rating"].isna().astype(int)
    m["customer_rating"] = m["customer_rating"].fillna(m["customer_rating"].median())
    # --- customers with no social profile: flag + neutral imputation
    m["has_social_profile"] = m["n_social_profiles"].notna().astype(int)
    num_social = ["n_social_profiles", "n_platforms", "avg_engagement", "max_engagement",
                  "avg_purchase_interest", "max_purchase_interest", "avg_sentiment"]
    for c in num_social:
        fill = 0 if c in ("n_social_profiles", "n_platforms") else m[c].median()
        m[c] = m[c].fillna(fill)
    for c in [c for c in m.columns if c.startswith("platform_share_")]:
        m[c] = m[c].fillna(0)
    m["top_platform"] = m["top_platform"].fillna("Unknown")
    # --- time features
    m["purchase_month"] = m["purchase_date"].dt.month
    m["purchase_dayofweek"] = m["purchase_date"].dt.dayofweek
    m["is_weekend"] = (m["purchase_dayofweek"] >= 5).astype(int)
    # --- behavioural features (do NOT use product_category -> no target leakage)
    m = m.sort_values(["customer_id", "purchase_date"]).reset_index(drop=True)
    m["customer_txn_count"] = m.groupby("customer_id")["transaction_id"].transform("count")
    m["customer_avg_amount"] = m.groupby("customer_id")["purchase_amount"].transform("mean").round(2)
    m["amount_vs_customer_avg"] = (m["purchase_amount"] / m["customer_avg_amount"]).round(3)
    m["amount_bucket"] = pd.qcut(m["purchase_amount"], 3, labels=["low", "medium", "high"]).astype(str)
    m["engagement_x_interest"] = (m["avg_engagement"] * m["avg_purchase_interest"]).round(2)
    for c in ("avg_engagement", "avg_purchase_interest", "avg_sentiment"):
        m[c] = m[c].round(3)
    return m


def one_hot(m: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(m, columns=["top_platform", "amount_bucket"], prefix=["top_platform", "amount"], dtype=int)


def main():
    social_raw, trans_raw = load_raw()
    social = clean_social(social_raw)
    trans = clean_transactions(trans_raw)
    social_agg = aggregate_social(social)

    # ---- merge: LEFT join, transactions are the base (they hold the target label)
    merged = trans.merge(social_agg, on="customer_id", how="left", validate="many_to_one")

    # ---- post-merge validation
    assert len(merged) == len(trans), "join changed number of transactions!"
    assert merged["transaction_id"].is_unique
    print(f"social raw rows: {len(social_raw)} -> after de-dup: {len(social)} ({len(social_raw)-len(social)} removed)")
    print(f"unique customers  social: {social['customer_id'].nunique()} | transactions: {trans['customer_id'].nunique()}")
    print(f"transactions with a social match: {merged['n_social_profiles'].notna().sum()} / {len(merged)}")

    final = engineer(merged)
    final_encoded = one_hot(final)
    assert final_encoded.drop(columns="purchase_date").isna().sum().sum() == 0, "NaNs remain"

    out = DATA / "merged_dataset.csv"
    final_encoded.to_csv(out, index=False)
    final.to_csv(DATA / "merged_dataset_readable.csv", index=False)
    print(f"saved {out}  shape={final_encoded.shape}")
    print(final_encoded["product_category"].value_counts())


if __name__ == "__main__":
    main()
