import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import IsolationForest

class IsoForestClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=100, contamination=0.01, random_state=42):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state

    def fit(self, X, y=None):
        X, y = check_X_y(X, y) if y is not None else (check_array(X), None)
        
        self.model_ = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state
        )
        self.model_.fit(X)
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        raw_preds = self.model_.predict(X)
        # Map: 1 = Normal and it becomes (0 means non frauds) and Anomaly(-1) becomes as 1.
        return np.where(raw_preds == -1, 1, 0)

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)
        # Use decision_function (lower is more anomalous)
        scores = self.model_.decision_function(X)
        # Convert to [0, 1] probability range; lower score = higher fraud prob
        # Professional standard: use a soft mapping or robust scaling
        prob_fraud = (scores.max() - scores) / (scores.max() - scores.min() + 1e-9)
        return np.vstack([1 - prob_fraud, prob_fraud]).T

# ---------- CACHED LOADERS ----------
@st.cache_resource
def load_models():
    try:
        hybrid_lg = joblib.load("hybrid_logistic_regression_model.pkl")
        hybrid_cb = joblib.load("hybrid_catboost_model.pkl")
        cbc = joblib.load("catboost_model.pkl")
        brf = joblib.load("balanced_random_forest_model_tuned.pkl")
        feature_names = joblib.load("fraud_features_names.pkl")
        thresholds = joblib.load("model_thresholds.pkl")  # dict with keys like 'hybridlogistic', 'hybridcatboost', 'catboost', 'balancedrf'
        return hybrid_lg, hybrid_cb, cbc, brf, feature_names, thresholds
    except ModuleNotFoundError as e:
        missing_pkg = getattr(e, "name", None) or str(e)
        st.error(
            "A required dependency is missing when loading the saved models.\n"
            f"Missing module: **{missing_pkg}**\n\n"
            f"Install it (e.g. `pip install {missing_pkg}`) and restart the app."
        )
        st.stop()
    except FileNotFoundError as e:
        st.error(
            "Could not find one or more model artifact files.\n"
            "Make sure the following files exist in the app folder:\n"
            "  - hybrid_logistic_regression_model.pkl\n"
            "  - hybrid_catboost_model.pkl\n"
            "  - catboost_model.pkl\n"
            "  - balanced_random_forest_model_tuned.pkl\n"
            "  - fraud_features_names.pkl\n"
            "  - model_thresholds.pkl\n"
            f"\nError: {e}"
        )
        st.stop()
    except Exception as e:
        st.error(f"Failed to load model artifacts: {e}")
        st.stop()

# ---------- PREDICTION UTILS ----------
def apply_threshold(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    """probabilities is 1D array of P(fraud=1); return 0/1 labels."""
    return (probabilities >= threshold).astype(int)

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features from raw transaction data to match the 28-feature space."""
    df = df.copy()

    # Time-based feature
    df['hour'] = df['step'] % 24

    # Balance error features
    df['orig_balance_error'] = df['newbalanceOrig'] + df['amount'] - df['oldbalanceOrg']
    df['dest_balance_error'] = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']

    # Mule account detection (accounts appearing multiple times)
    df['is_mule_orig'] = df.groupby('nameOrig')['nameOrig'].transform('count') > 1
    df['is_mule_dest'] = df.groupby('nameDest')['nameDest'].transform('count') > 1

    # Transfer to zero destination
    df['transfer_to_zero_dest'] = (df['newbalanceDest'] == 0).astype(int)

    # Amount to original balance ratio (avoid div by zero)
    df['amount_to_orig_ratio'] = df['amount'] / df['oldbalanceOrg'].replace(0, np.inf)
    df['amount_to_orig_ratio'] = df['amount_to_orig_ratio'].replace(np.inf, 0)

    # Suspicious pattern: amount > original balance
    df['is_suspicious_pattern'] = (df['amount'] > df['oldbalanceOrg']).astype(int)

    # Log transformations
    df['amount_log'] = np.log(df['amount'] + 1)
    df['oldbalanceOrg_log'] = np.log(df['oldbalanceOrg'] + 1)
    df['newbalanceOrig_log'] = np.log(df['newbalanceOrig'] + 1)
    df['oldbalanceDest_log'] = np.log(df['oldbalanceDest'] + 1)
    df['newbalanceDest_log'] = np.log(df['newbalanceDest'] + 1)

    # Outlier flags (simple: above 95th percentile)
    for col in ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']:
        q95 = df[col].quantile(0.95)
        df[f'{col}_outlier'] = (df[col] > q95).astype(int)

    # Type one-hot encoding (baseline: CASH_IN, so dummies for others)
    type_dummies = pd.get_dummies(df['type'], prefix='type')
    if 'type_CASH_IN' in type_dummies.columns:
        type_dummies = type_dummies.drop('type_CASH_IN', axis=1)
    df = pd.concat([df, type_dummies], axis=1)

    return df

def prepare_features(df: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    """Select and order columns to match training feature space."""
    missing = [col for col in feature_names if col not in df.columns]
    extra = [col for col in df.columns if col not in feature_names]

    if missing:
        st.error(
            "Input data is missing required engineered columns:\n"
            + ", ".join(missing)
        )
        st.stop()

    # Warn about extra columns (safe to ignore)
    if extra:
        st.info(
            "Ignoring extra columns not used by the model:\n"
            + ", ".join(extra)
        )

    return df[feature_names].copy()

def main():
    st.set_page_config(page_title="JP Morgan Fraud Detection – Streamlit", layout="wide")
    st.title("JP Morgan Fraud Detection – Streamlit Deployment")

    st.markdown(
        """
        This app serves multiple trained fraud detection models (Hybrid Logistic Regression, Hybrid CatBoost, CatBoost, and Balanced Random Forest)  
        on 28 engineered features.
        """
    )

    # Load models & metadata
    with st.spinner("Loading models and metadata..."):
        model_hybrid_lg, model_hybrid_cb, model_cb, model_brf, feature_names, thresholds = load_models()

    # ------------- FILE UPLOAD -------------
    st.header("1. Upload transaction data")
    uploaded_file = st.file_uploader(
        "Upload CSV or Parquet file with feature columns",
        type=["csv", "parquet"]
    )

    if uploaded_file is None:
        st.info("Please upload a CSV/Parquet file to continue.")
        return

    # Detect type and read
    try:
        if uploaded_file.name.lower().endswith(".csv"):
            raw_df = pd.read_csv(uploaded_file)
        else:
            raw_df = pd.read_parquet(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        return

    # Show basic info
    st.success(f"✅ Loaded {raw_df.shape[0]:,} rows and {raw_df.shape[1]:,} columns.")
    with st.expander("Preview uploaded data"):
        st.dataframe(raw_df.head())

    # ------------- FEATURE ENGINEERING -------------
    st.header("2. Engineer features")
    st.write("Transforming raw data into the 28 engineered features required by the model.")

    with st.spinner("Engineering features..."):
        engineered_df = engineer_features(raw_df)

    st.success(f"✅ Feature engineering complete. Data now has {engineered_df.shape[1]} columns.")
    with st.expander("Preview engineered data"):
        st.dataframe(engineered_df.head())

    # ------------- FEATURE PREP -------------
    st.header("3. Prepare features")
    st.write("Selecting and ordering features to match the model's training space.")

    features_df = prepare_features(engineered_df, feature_names)

    st.success(f"Feature matrix ready with shape: {features_df.shape[0]:,} × {features_df.shape[1]}")
    with st.expander("Show feature matrix preview"):
        st.dataframe(features_df.head())

    # ------------- MODEL & THRESHOLD -------------
    st.header("4. Choose model and threshold")

    model_choice = st.selectbox(
        "Select model",
        [
            "Hybrid Logistic Regression",
            "Hybrid CatBoost",
            "CatBoost",
            "Balanced Random Forest",
        ],
        index=0,
    )

    if model_choice == "Hybrid Logistic Regression":
        model_key = "hybridlogistic"
        model = model_hybrid_lg
    elif model_choice == "Hybrid CatBoost":
        model_key = "hybridcatboost"
        model = model_hybrid_cb
    elif model_choice == "CatBoost":
        model_key = "catboost"
        model = model_cb
    else:
        model_key = "balancedrf"
        model = model_brf

    default_threshold = float(thresholds.get(model_key, 0.5))

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        threshold = st.slider(
            "Decision threshold (probability ≥ threshold ⇒ Fraud=1)",
            min_value=0.0,
            max_value=1.0,
            value=default_threshold,
            step=0.01,
        )
    with col_t2:
        st.write(f"Default threshold from training: **{default_threshold:.2f}**")

    # Optional: compare models on this uploaded dataset
    if st.button("Compare all models on this dataset"):
        if "isFraud" not in raw_df.columns:
            st.error("Comparison requires the `isFraud` column to be present in the uploaded data.")
        else:
            y_true = raw_df["isFraud"].astype(int)

            def compute_metrics(model, key):
                th = float(thresholds.get(key, 0.5))
                probs = model.predict_proba(features_df)[:, 1]
                preds = (probs >= th).astype(int)
                tn = int(((y_true == 0) & (preds == 0)).sum())
                fp = int(((y_true == 0) & (preds == 1)).sum())
                fn = int(((y_true == 1) & (preds == 0)).sum())
                tp = int(((y_true == 1) & (preds == 1)).sum())
                recall = tp / (tp + fn) if (tp + fn) > 0 else None
                precision = tp / (tp + fp) if (tp + fp) > 0 else None
                acc = (tp + tn) / len(y_true)
                return {
                    "model": key,
                    "threshold": th,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "tn": tn,
                    "precision": precision,
                    "recall": recall,
                    "accuracy": acc,
                }

            comparisons = []
            comparisons.append(compute_metrics(model_hybrid_lg, "hybridlogistic"))
            comparisons.append(compute_metrics(model_hybrid_cb, "hybridcatboost"))
            comparisons.append(compute_metrics(model_cb, "catboost"))
            comparisons.append(compute_metrics(model_brf, "balancedrf"))

            comp_df = pd.DataFrame(comparisons)
            st.write("### Model comparison")
            st.dataframe(comp_df)

    # Optional: find an optimal threshold that limits False Negatives
    with st.expander("Find optimal threshold (minimize FN while controlling FP)"):
        max_fn = st.number_input(
            "Maximum allowed false negatives (missed fraud)",
            min_value=0,
            max_value=50,
            value=0,
            step=1,
        )

        if st.button("Scan thresholds"):
            with st.spinner("Scanning thresholds for best tradeoff..."):
                try:
                    y_true = raw_df["isFraud"].astype(int)
                    y_probs = model.predict_proba(features_df)[:, 1]
                except Exception as e:
                    st.error(f"Model prediction failed: {e}")
                    y_probs = None

                if y_probs is not None:
                    def scan_thresholds(probas, y_true, start=0.5, end=0.96, step=0.05, max_fn=0):
                        rows = []
                        thresholds = np.arange(start, end + 1e-9, step)
                        for t in thresholds:
                            y_pred = (probas >= t).astype(int)
                            tn = int(((y_true == 0) & (y_pred == 0)).sum())
                            fp = int(((y_true == 0) & (y_pred == 1)).sum())
                            fn = int(((y_true == 1) & (y_pred == 0)).sum())
                            tp = int(((y_true == 1) & (y_pred == 1)).sum())
                            status = "STABLE" if fn <= max_fn else "LOSING FRAUD!"
                            rows.append({
                                "threshold": t,
                                "tn": tn,
                                "fp": fp,
                                "fn": fn,
                                "tp": tp,
                                "status": status,
                            })
                        return pd.DataFrame(rows)

                    result_df = scan_thresholds(y_probs, y_true, max_fn=max_fn)
                    st.dataframe(result_df)

    # ------------- RUN INFERENCE -------------
    st.header("5. Run prediction")

    if st.button("Predict fraud for all rows"):
        with st.spinner("Running model inference..."):
            try:
                proba = model.predict_proba(features_df)[:, 1]
            except Exception as e:
                st.error(f"Model prediction failed: {e}")
                return

        labels = apply_threshold(proba, threshold)

        # Build result DataFrame
        result_df = raw_df.copy()
        result_df["fraud_probability"] = proba
        result_df["fraud_predicted"] = labels

        # ------------- RISK TIERS (block / review / allow) -------------
        with st.expander("Tiered scoring (block / review / allow)"):
            col_a, col_b = st.columns(2)
            with col_a:
                block_threshold = st.slider(
                    "Block if score ≥", 0.0, 1.0, 0.90, 0.01,
                    help="Transactions with score ≥ this value are treated as high-risk (block)."
                )
            with col_b:
                review_threshold = st.slider(
                    "Review if score ≥", 0.0, 1.0, 0.60, 0.01,
                    help="Transactions with score ≥ this value (but below block threshold) are flagged for human review."
                )

            if review_threshold > block_threshold:
                st.warning("Review threshold should be lower than or equal to the block threshold for a consistent tiering.")

            def _action_from_score(p):
                if p >= block_threshold:
                    return "Block"
                if p >= review_threshold:
                    return "Review"
                return "Allow"

            result_df["risk_action"] = result_df["fraud_probability"].apply(_action_from_score)

            st.write("### Tier breakdown")
            tier_counts = result_df["risk_action"].value_counts().reindex(["Block", "Review", "Allow"]).fillna(0).astype(int)
            st.write(tier_counts.to_frame(name="count"))

            if "isFraud" in result_df.columns:
                # Show how many frauds fall into each tier (help identify miss rate in "Allow")
                tier_frauds = result_df[result_df["isFraud"] == 1]["risk_action"].value_counts().reindex(["Block", "Review", "Allow"]).fillna(0).astype(int)
                st.write("### Frauds by tier")
                st.write(tier_frauds.to_frame(name="fraud_count"))

        # ------------- SUMMARY METRICS -------------
        total_rows = len(result_df)
        predicted_fraud = int(labels.sum())
        predicted_normal = int(total_rows - predicted_fraud)

        # Ground truth / performance metrics (if available)
        actual_fraud = None
        fraud_detected = None
        fraud_recall = None
        overall_accuracy = None
        tn = fp = fn = tp = None

        if "isFraud" in raw_df.columns:
            y_true = raw_df["isFraud"].astype(int)
            y_pred = labels.astype(int)

            actual_fraud = int(y_true.sum())
            fraud_detected = int(((y_true == 1) & (y_pred == 1)).sum())
            fraud_recall = (fraud_detected / actual_fraud) if actual_fraud > 0 else None
            overall_accuracy = ((y_true == y_pred).sum() / total_rows) if total_rows > 0 else None

            tp = int(((y_true == 1) & (y_pred == 1)).sum())
            tn = int(((y_true == 0) & (y_pred == 0)).sum())
            fp = int(((y_true == 0) & (y_pred == 1)).sum())
            fn = int(((y_true == 1) & (y_pred == 0)).sum())

        st.subheader("Summary")
        st.write(f"Total rows: **{total_rows:,}**")

        if actual_fraud is not None:
            st.write(f"Actual fraud (isFraud=1): **{actual_fraud:,}**")
            st.write(f"Fraud detected by model: **{fraud_detected:,}**")
            if fraud_recall is not None:
                st.write(f"Fraud detection rate (recall): **{fraud_recall * 100:.1f}%**")
            if overall_accuracy is not None:
                st.write(f"Overall accuracy: **{overall_accuracy * 100:.1f}%**")

            st.markdown("**Confusion matrix (actual × predicted)**")
            st.markdown(
                "| | Pred = 0 | Pred = 1 |\n"
                "|---|---|---|\n"
                f"| Actual = 0 | {tn:,} | {fp:,} |\n"
                f"| Actual = 1 | {fn:,} | {tp:,} |\n"
            )

        st.write(f"Predicted fraud (1): **{predicted_fraud:,}**")
        st.write(f"Predicted non‑fraud (0): **{predicted_normal:,}**")

        with st.expander("Show predictions (first 100 rows)"):
            st.dataframe(result_df.head(100))

        # Download
        csv_bytes = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download predictions as CSV",
            data=csv_bytes,
            file_name=f"fraud_predictions_{model_key}.csv",
            mime="text/csv",
        )

if __name__ == "__main__":
    main()
