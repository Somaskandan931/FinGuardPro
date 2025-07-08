from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import traceback
import pickle
import joblib
import os

from modules.transaction_screening import run_rule_engine
from modules.name_screening import run_name_screening
from modules.decision_engine import combine_decision
from explain.explain_utils import explain_transaction, explain_summary_plot
from auth.auth import login, token_required
from reports.report_utils import generate_fraud_report

app = Flask(__name__)

# ------------------- MODEL PATHS ----------------------

BASE_MODEL_PATH = "C:/Users/somas/PycharmProjects/FinGuardPro/models"
MODEL_PATH = os.path.join(BASE_MODEL_PATH, "xgboost_tuned_model.pkl")
SCALER_PATH = os.path.join(BASE_MODEL_PATH, "scaler.pkl")
ENCODERS_PATH = os.path.join(BASE_MODEL_PATH, "label_encoders.pkl")

# ------------------- OUTPUT PATHS ----------------------

OUTPUT_BASE_DIR = "C:/Users/somas/PycharmProjects/FinGuardPro/output"
SHAP_OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, "shap_outputs")
REPORT_OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, "reports")

os.makedirs(SHAP_OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_OUTPUT_DIR, exist_ok=True)

# ------------------- LOAD MODEL COMPONENTS ----------------------

try:
    model = joblib.load(MODEL_PATH)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(ENCODERS_PATH, "rb") as f:
        label_encoders = pickle.load(f)
    categorical_columns = list(label_encoders.keys())

    # Hardcoded feature columns (must exactly match scaler & model training)
    feature_columns = [
        'sender_balance_before',
        'sender_age',
        'sender_suspicious_name',
        'recipient_balance_before',
        'recipient_suspicious_name',
        'transaction_amount',
        'hour_of_day',
        'day_of_week',
        'is_weekend',
        'txns_last_hour',
        'txns_last_day',
        'txns_last_week',
        'amount_to_balance_ratio',
        'amount_vs_channel_limit_ratio',
        'is_round_amount',
        'hour',
        'is_night',
        'log_amount',
        'is_high_value',
        'is_new_receiver',
        'sender_txn_count',
        'amount_to_avg_ratio',
        'sender_id_encoded',
        'sender_account_type_encoded',
        'sender_risk_profile_encoded',
        'recipient_id_encoded',
        'recipient_account_type_encoded',
        'transaction_type_encoded',
        'device_type_encoded',
        'location_encoded',
        'merchant_category_encoded'
    ]

    print(f"✅ Loaded model components with {len(feature_columns)} feature columns")
except Exception as e:
    print(f"❌ Error loading model components: {e}")
    raise

EXCLUDE_FROM_MODEL = ['is_impossible_travel', 'is_round_trip', 'round_trip_chain_id', 'round_trip_position']

# ------------------- HELPERS ----------------------

def safe_label_encode(series, le):
    def encode_val(x):
        if pd.isnull(x):
            return -1
        try:
            return le.transform([x])[0] if x in le.classes_ else -1
        except Exception:
            return -1
    return series.apply(encode_val)

def ensure_features_exist(df, feature_cols):
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    return df

def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Encode categorical columns safely
    for col in categorical_columns:
        if col in df.columns:
            le = label_encoders[col]
            df[col + "_encoded"] = safe_label_encode(df[col], le)
        else:
            df[col + "_encoded"] = -1

    # Drop original categorical columns
    df.drop(columns=categorical_columns, inplace=True, errors='ignore')

    # Drop excluded columns not used for model input
    df.drop(columns=EXCLUDE_FROM_MODEL, inplace=True, errors='ignore')

    # Add missing features as zeros
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    # Reorder columns to match exactly the feature_columns order
    df = df[feature_columns]

    # Scale features (expects exact columns and order)
    X_scaled = scaler.transform(df)
    return X_scaled

def determine_risk_level(score: float) -> str:
    if score >= 0.8:
        return "High"
    elif score >= 0.5:
        return "Medium"
    else:
        return "Low"

# ------------------- ROUTES ----------------------

@app.route("/login", methods=["POST"])
def handle_login():
    return login()

@app.route("/predict", methods=["POST"])
@token_required
def predict():
    try:
        input_json = request.get_json()
        if not input_json:
            return jsonify({"error": "Empty or invalid input."}), 400

        # Create DataFrame from input JSON
        transactions_df = pd.DataFrame([input_json]) if isinstance(input_json, dict) else pd.DataFrame(input_json)

        # Ensure all required features exist (including excluded for full logic)
        transactions_df = ensure_features_exist(transactions_df, feature_columns + EXCLUDE_FROM_MODEL)

        # Run rule engine and name screening
        rules = run_rule_engine(transactions_df)
        names = run_name_screening(transactions_df)

        # Run model prediction with error handling
        try:
            X = preprocess_features(transactions_df)
            preds = model.predict_proba(X)[:, 1]
        except Exception as e:
            print(f"❌ Model prediction error: {e}")
            preds = [0.0] * len(transactions_df)

        # Ensure transaction_id present for reporting
        if "transaction_id" not in transactions_df.columns:
            transactions_df["transaction_id"] = [f"tx_{i}" for i in range(len(transactions_df))]

        # Prepare model predictions DataFrame
        model_preds = pd.DataFrame({
            "transaction_id": transactions_df["transaction_id"],
            "fraud_score": preds
        })

        # Combine decisions (model + rules + name screening)
        final_df = combine_decision(transactions_df, model_preds, rules, names)

        responses = []
        for idx, row in final_df.iterrows():
            txn_data = transactions_df.iloc[idx].to_dict()
            fraud_score = row.get("fraud_score", 0.0)
            risk_level = determine_risk_level(fraud_score)

            # Generate PDF fraud report
            report_path = generate_fraud_report(
                txn_data=txn_data,
                fraud_score=fraud_score,
                risk_level=risk_level,
                save_path=os.path.join(REPORT_OUTPUT_DIR, f"fraud_report_{txn_data.get('transaction_id', idx)}.pdf")
            )

            # Generate SHAP explanation plots
            shap_paths = explain_transaction(
                model=model,
                input_df=transactions_df.iloc[[idx]],
                scaler=scaler,
                encoders=label_encoders,
                feature_columns=feature_columns,
                transaction_ids=[txn_data.get("transaction_id", f"tx_{idx}")],
                output_dir=SHAP_OUTPUT_DIR
            )

            row_dict = row.to_dict()
            row_dict["risk_level"] = risk_level
            row_dict["report_path"] = f"/reports/fraud_report_{txn_data.get('transaction_id', idx)}.pdf"
            row_dict["shap_plot"] = f"/explain/shap_outputs/{os.path.basename(shap_paths[0])}" if shap_paths else None
            responses.append(row_dict)

        # If batch prediction, include SHAP summary plot
        if len(transactions_df) > 1:
            summary_path = explain_summary_plot(
                model=model,
                input_df=transactions_df,
                scaler=scaler,
                encoders=label_encoders,
                feature_columns=feature_columns,
                output_path=os.path.join(OUTPUT_BASE_DIR, "shap_summary.png")
            )
            return jsonify({
                "results": responses,
                "shap_summary_plot": "/output/shap_summary.png"
            }), 200

        # Single prediction response
        return jsonify(responses[0] if len(responses) == 1 else responses), 200

    except Exception as e:
        print("❌ Error in /predict:", traceback.format_exc())
        return jsonify({"error": "Internal server error", "details": str(e)}), 500


@app.route('/explain/shap_outputs/<path:filename>')
def serve_shap_file(filename):
    return send_from_directory(SHAP_OUTPUT_DIR, filename)


@app.route('/reports/<path:filename>')
def serve_report_file(filename):
    return send_from_directory(REPORT_OUTPUT_DIR, filename)


if __name__ == "__main__":
    app.run(debug=True)
