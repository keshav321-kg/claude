"""Synthetic financial-transaction data generation.

Real fraud datasets (e.g. credit-card transactions) are highly imbalanced
and privacy-sensitive, so this module produces a synthetic stand-in with
similar structure: mostly legitimate transactions plus a small fraction of
fraud with distinct statistical signatures (odd hours, high amounts,
location mismatches, rapid repeat transactions).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "amount",
    "hour",
    "distance_from_home_km",
    "distance_from_last_txn_km",
    "txns_last_hour",
    "is_foreign_country",
    "is_new_merchant",
    "avg_amount_ratio",
]


def generate_transactions(n_samples: int = 50_000, fraud_ratio: float = 0.015, random_state: int = 42) -> pd.DataFrame:
    """Generate a synthetic labeled transaction dataset.

    Returns a DataFrame with FEATURE_COLUMNS plus an `is_fraud` label column.
    """
    rng = np.random.default_rng(random_state)
    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud

    legit = pd.DataFrame(
        {
            "amount": rng.gamma(shape=2.0, scale=40, size=n_legit),
            "hour": rng.normal(loc=14, scale=4, size=n_legit) % 24,
            "distance_from_home_km": rng.exponential(scale=5, size=n_legit),
            "distance_from_last_txn_km": rng.exponential(scale=3, size=n_legit),
            "txns_last_hour": rng.poisson(lam=0.5, size=n_legit),
            "is_foreign_country": rng.binomial(1, 0.02, size=n_legit),
            "is_new_merchant": rng.binomial(1, 0.1, size=n_legit),
            "avg_amount_ratio": rng.normal(loc=1.0, scale=0.3, size=n_legit).clip(0.1),
        }
    )
    legit["is_fraud"] = 0

    fraud = pd.DataFrame(
        {
            "amount": rng.gamma(shape=3.0, scale=250, size=n_fraud),
            "hour": rng.normal(loc=3, scale=3, size=n_fraud) % 24,
            "distance_from_home_km": rng.exponential(scale=400, size=n_fraud),
            "distance_from_last_txn_km": rng.exponential(scale=300, size=n_fraud),
            "txns_last_hour": rng.poisson(lam=4, size=n_fraud),
            "is_foreign_country": rng.binomial(1, 0.55, size=n_fraud),
            "is_new_merchant": rng.binomial(1, 0.7, size=n_fraud),
            "avg_amount_ratio": rng.normal(loc=6.0, scale=3.0, size=n_fraud).clip(0.5),
        }
    )
    fraud["is_fraud"] = 1

    df = pd.concat([legit, fraud], ignore_index=True)
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    df["amount"] = df["amount"].round(2)
    df["hour"] = df["hour"].round(2)
    return df
