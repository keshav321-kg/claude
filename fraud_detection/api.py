"""FastAPI service that scores transactions for fraud risk.

Run with:
    uvicorn fraud_detection.api:app --reload
"""
from __future__ import annotations

import os
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from fraud_detection.data import FEATURE_COLUMNS

MODEL_PATH = Path(os.environ.get("FRAUD_MODEL_PATH", "artifacts/model.joblib"))
DEFAULT_THRESHOLD = float(os.environ.get("FRAUD_THRESHOLD", "0.5"))

app = FastAPI(title="Fraud Detection API", version="1.0.0")
_model = None


class Transaction(BaseModel):
    amount: float = Field(..., ge=0, description="Transaction amount")
    hour: float = Field(..., ge=0, lt=24, description="Hour of day the transaction occurred")
    distance_from_home_km: float = Field(..., ge=0)
    distance_from_last_txn_km: float = Field(..., ge=0)
    txns_last_hour: int = Field(..., ge=0)
    is_foreign_country: int = Field(..., ge=0, le=1)
    is_new_merchant: int = Field(..., ge=0, le=1)
    avg_amount_ratio: float = Field(..., gt=0, description="Amount / account's historical average amount")


class ScoreResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool
    threshold: float


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail=f"Model not found at {MODEL_PATH}. Train it first with `python -m fraud_detection.train`.",
            )
        _model = joblib.load(MODEL_PATH)
    return _model


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}


@app.post("/score", response_model=ScoreResponse)
def score(txn: Transaction, threshold: float = DEFAULT_THRESHOLD) -> ScoreResponse:
    model = get_model()
    row = [[getattr(txn, col) for col in FEATURE_COLUMNS]]
    proba = float(model.predict_proba(row)[0, 1])
    return ScoreResponse(fraud_probability=proba, is_fraud=proba >= threshold, threshold=threshold)
