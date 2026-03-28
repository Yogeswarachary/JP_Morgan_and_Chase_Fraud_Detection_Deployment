# JPMorgan & Chase Fraud Detection — Streamlit App

## About this repository

This project is a simple, interactive fraud detection deployment built with Streamlit. It loads trained models and crawls a raw transaction dataset into engineered features, then predicts fraud risk with multiple model options.

Key functionality:
- 4 fraud models supported:
  - Hybrid Logistic Regression
  - Hybrid CatBoost
  - CatBoost
  - Balanced Random Forest
- Automatic feature engineering from raw transaction input
- Threshold tuning and model comparison
- Tiered risk actions (`Block`, `Review`, `Allow`)
- Prediction download as CSV
- Optional performance metrics when `isFraud` label is present

## What the app does (from `app.py`)

1. Load model artifacts and thresholds
2. Upload transaction data (CSV / Parquet)
3. Generate 28 engineered features (balance errors, account reuse, log transforms, outlier flags, type dummies, etc.)
4. Prepare a feature matrix exactly matching model training columns
5. Select model + threshold and run inference
6. Show predictions and risk tiers
7. Provide optional threshold scanning and model comparison
8. Download results (`fraud_probability`, `fraud_predicted`, `risk_action`)

## Getting started

### Prerequisites

- Python 3.9+ (3.10 / 3.11 recommended)
- `pip`
- Streamlit environment

### 1) Clone repository

```bash
git clone https://github.com/<your-org>/JP_Morgan_and_Chase_Fraud_Detection_Deployment.git
cd JP_Morgan_and_Chase_Fraud_Detection_Deployment
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Place required model artifact files in repository root

- `hybrid_logistic_regression_model.pkl`
- `hybrid_catboost_model.pkl`
- `catboost_model.pkl`
- `balanced_random_forest_model_tuned.pkl`
- `fraud_features_names.pkl`
- `model_thresholds.pkl`

### 4) Run app locally

```bash
streamlit run app.py
```

### 5) Open in browser

Visit the link shown in terminal (typically `http://localhost:8501`).

## Input data format

- Required raw columns (from bank transaction data include but not limited to):
  - `step`, `type`, `amount`, `nameOrig`, `oldbalanceOrg`, `newbalanceOrig`, `nameDest`, `oldbalanceDest`, `newbalanceDest`, optionally `isFraud` for evaluation.
- App computes extra features and then uses model input features list from `fraud_features_names.pkl`.

## File summary

- `app.py` — main Streamlit app and model pipeline
- `requirements.txt` — required Python packages
- saved model files (external artifacts) as listed above

## Deployment notes

- For Streamlit Cloud, set main file to `app.py`.
- Keep artifact files in workspace root (or update paths in `load_models()` if moved).

---

If you want, I can also add a short sample input CSV snippet and quick Docker instructions next.
