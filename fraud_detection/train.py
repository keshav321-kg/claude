"""Train and evaluate a fraud-detection classifier.

Usage:
    python -m fraud_detection.train --model-path artifacts/model.joblib
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fraud_detection.data import FEATURE_COLUMNS, generate_transactions


def build_pipeline(random_state: int = 42) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=12,
                    min_samples_leaf=5,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=random_state,
                ),
            ),
        ]
    )


def train(n_samples: int = 50_000, random_state: int = 42) -> tuple[Pipeline, dict]:
    df = generate_transactions(n_samples=n_samples, random_state=random_state)
    X = df[FEATURE_COLUMNS]
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=random_state
    )

    pipeline = build_pipeline(random_state=random_state)
    pipeline.fit(X_train, y_train)

    proba = pipeline.predict_proba(X_test)[:, 1]
    preds = pipeline.predict(X_test)

    metrics = {
        "roc_auc": roc_auc_score(y_test, proba),
        "average_precision": average_precision_score(y_test, proba),
        "report": classification_report(y_test, preds, digits=3),
    }
    return pipeline, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=Path("artifacts/model.joblib"))
    parser.add_argument("--n-samples", type=int, default=50_000)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    pipeline, metrics = train(n_samples=args.n_samples, random_state=args.random_state)

    print(f"ROC AUC:           {metrics['roc_auc']:.4f}")
    print(f"Average precision: {metrics['average_precision']:.4f}")
    print(metrics["report"])

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, args.model_path)
    print(f"Model saved to {args.model_path}")


if __name__ == "__main__":
    main()
