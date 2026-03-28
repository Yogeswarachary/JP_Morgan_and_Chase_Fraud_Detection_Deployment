# Fraud Detection Streamlit App

This repository contains a Streamlit-based deployment for a credit card fraud detection model.

## Getting Started

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Run locally

```bash
streamlit run app.py
```

### 3) Deploy on Streamlit Cloud

1. Push this repository to a GitHub repo.
2. Create a new app in Streamlit Cloud and point it at this repo.
3. Set the **Main file** to `app.py` if needed.

## Files

- `app.py`: Streamlit app logic (data upload, feature engineering, model inference, download results).
- `requirements.txt`: Python dependencies.

## Notes

- This app performs **automatic feature engineering** on uploaded raw transaction data (11 columns) to create the 28 features required by the models.
- The app depends on the `catboost` package because one or more saved models were trained using CatBoost. It is listed in `requirements.txt`.
- Place model artifacts in the repository root:
  - `hybrid_logistic_regression_model.pkl`
  - `balanced_random_forest_model_tuned.pkl`
  - `fraud_features_names.pkl`
  - `model_thresholds.pkl`

