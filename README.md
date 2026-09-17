# Fraud Detection

A financial-transaction fraud detection system: a synthetic data generator, a
scikit-learn training pipeline (`RandomForestClassifier` on a standardized
feature set), and a FastAPI service that scores transactions for fraud risk
in real time.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/keshav321-kg/claude/blob/main/notebooks/fraud_detection_colab.ipynb)

A self-contained notebook version (`notebooks/fraud_detection_colab.ipynb`)
lets you generate the data, train, and score example transactions directly
in Google Colab — click the badge above.

## Features used

- `amount` — transaction amount
- `hour` — hour of day
- `distance_from_home_km` — distance from the account's home location
- `distance_from_last_txn_km` — distance from the previous transaction
- `txns_last_hour` — transaction velocity
- `is_foreign_country` — foreign-country flag
- `is_new_merchant` — first-time merchant flag
- `avg_amount_ratio` — amount relative to the account's historical average

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Train the model

```bash
python -m fraud_detection.train --model-path artifacts/model.joblib
```

Prints ROC AUC, average precision, and a classification report, then saves
the trained pipeline to `artifacts/model.joblib`.

## Serve predictions

```bash
export FRAUD_MODEL_PATH=artifacts/model.joblib
uvicorn fraud_detection.api:app --reload
```

- `GET /health` — service and model status
- `POST /score` — score a transaction, e.g.:

```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{"amount": 4000, "hour": 3, "distance_from_home_km": 900,
       "distance_from_last_txn_km": 700, "txns_last_hour": 6,
       "is_foreign_country": 1, "is_new_merchant": 1, "avg_amount_ratio": 9.0}'
```

Response:

```json
{"fraud_probability": 0.93, "is_fraud": true, "threshold": 0.5}
```

## Tests

```bash
pytest
```

## Notes

The bundled dataset is synthetic, generated to mimic realistic fraud
patterns (odd-hour transactions, large amounts, location jumps, high
velocity) without using any real financial data. Swap `fraud_detection/data.py`
for a loader against your own labeled transaction data to use this in
production, and re-evaluate feature engineering, class-imbalance handling,
and monitoring/drift checks accordingly.
