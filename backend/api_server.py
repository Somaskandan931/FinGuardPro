import numpy as np
from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import traceback
import pickle
import joblib
import os
from typing import List, Dict, Any

from sqlalchemy import create_engine, text

from modules.decision_engine import combine_decision
from modules.transaction_screening import run_transaction_rules
from modules.name_screening import run_name_screening, DB_CONFIG
from utils.explain_utils import explain_transaction, explain_summary_plot
from auth.auth import login, token_required
from reports.report_utils import generate_fraud_report
from utils.preprocessing import preprocess_features
import logging

app = Flask(__name__)

# Logging config
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------- CONSTANTS & PATHS -------------------

BASE_MODEL_PATH = "C:/Users/somas/PycharmProjects/FinGuardPro/models"
MODEL_PATH = os.path.join(BASE_MODEL_PATH, "xgboost_best_model.pkl")

SCALER_PATH = "C:/Users/somas/PycharmProjects/FinGuardPro/data/scaler.pkl"
ENCODERS_PATH = "C:/Users/somas/PycharmProjects/FinGuardPro/data/label_encoders.pkl"

OUTPUT_BASE_DIR = "C:/Users/somas/PycharmProjects/FinGuardPro/output"
SHAP_OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, "shap_outputs")
REPORT_OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, "reports")

os.makedirs(SHAP_OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_OUTPUT_DIR, exist_ok=True)

EXCLUDE_FROM_MODEL = [
    'is_impossible_travel', 'is_round_trip', 'round_trip_chain_id', 'round_trip_position'
]

try:
    model = joblib.load(MODEL_PATH)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(ENCODERS_PATH, "rb") as f:
        label_encoders: Dict[str, Any] = pickle.load(f)
    categorical_columns: List[str] = list(label_encoders.keys())

    feature_columns: List[str] = [
        "sender_balance_before",
        "sender_age",
        "recipient_balance_before",
        "transaction_type",
        "device_type",
        "location",
        "merchant_category",
        "amount",
        "hour_of_day",
        "day_of_week",
        "is_weekend",
        "txns_last_hour",
        "txns_last_day",
        "txns_last_week",
        "amount_to_balance_ratio",
        "amount_vs_channel_limit_ratio",
        "is_round_amount",
        "is_high_value",
        "log_amount",
        "is_new_receiver",
        "sender_txn_count",
        "amount_to_avg_ratio",
        "sender_account_type",
        "sender_risk_profile",
        "recipient_account_type"
    ]

    print(f"✅ Loaded model components with {len(feature_columns)} feature columns")

except Exception as e:
    print(f"❌ Error loading model components: {e}")
    raise


# ------------------- HELPERS -------------------

def ensure_features_exist(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    return df


def determine_risk_level(score: float) -> str:
    if score >= 0.8:
        return "High"
    elif score >= 0.5:
        return "Medium"
    return "Low"


# ------------------- ROUTES -------------------

@app.route("/login", methods=["POST"])
def handle_login():
    return login()


def safe_cast(val):
    if isinstance(val, (np.floating, np.float32, np.float64)):
        return float(val)
    elif isinstance(val, (np.integer, np.int32, np.int64)):
        return int(val)
    elif isinstance(val, (np.bool_)):
        return bool(val)
    return val

@app.route("/predict", methods=["POST"])
@token_required
def predict():
    try:
        input_json = request.get_json()
        if not input_json:
            return jsonify({"error": "Empty or invalid input."}), 400

        # Convert input JSON to DataFrame
        transactions_df = (
            pd.DataFrame([input_json])
            if isinstance(input_json, dict)
            else pd.DataFrame(input_json)
        )

        transactions_df = ensure_features_exist(transactions_df, feature_columns + EXCLUDE_FROM_MODEL)

        # Model prediction
        try:
            X = preprocess_features(
                df=transactions_df,
                label_encoders=label_encoders,
                scaler=scaler,
                categorical_columns=categorical_columns
            )
            preds = model.predict_proba(X)[:, 1]
        except Exception as e:
            print(f"❌ Model prediction error: {e}")
            preds = [0.0] * len(transactions_df)

        # Ensure transaction_id exists
        if "transaction_id" not in transactions_df.columns:
            transactions_df["transaction_id"] = [f"tx_{i}" for i in range(len(transactions_df))]

        responses = []

        for idx, row in transactions_df.iterrows():
            txn_data = row.to_dict()
            fraud_score = preds[idx]
            risk_level = determine_risk_level(fraud_score)

            # Generate PDF report
            report_path = generate_fraud_report(
                txn_data=txn_data,
                fraud_score=fraud_score,
                risk_level=risk_level,
                save_path=os.path.join(REPORT_OUTPUT_DIR, f"fraud_report_{txn_data['transaction_id']}.pdf")
            )

            # Generate SHAP explanation plots
            shap_paths = explain_transaction(
                model=model,
                input_df=transactions_df.iloc[[idx]],
                scaler=scaler,
                encoders=label_encoders,
                feature_columns=feature_columns,
                transaction_ids=[txn_data['transaction_id']],
                output_dir=SHAP_OUTPUT_DIR
            )

            response = {
                "transaction_id": txn_data["transaction_id"],
                "fraud_score": fraud_score,
                "risk_level": risk_level,
                "report_path": f"/reports/fraud_report_{txn_data['transaction_id']}.pdf",
                "shap_plot": f"/explain/shap_outputs/{os.path.basename(shap_paths[0])}" if shap_paths else None
            }

            # Convert all NumPy values to native types
            responses.append({k: safe_cast(v) for k, v in response.items()})

        # SHAP summary for batch predictions
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

        return jsonify(responses[0] if len(responses) == 1 else responses), 200

    except Exception:
        print("❌ Error in /predict:", traceback.format_exc())
        return jsonify({"error": "Internal server error"}), 500
@app.route("/names-screening", methods=["POST"])
def name_screening_route():
    try:
        input_json = request.get_json()
        logger.info(f"📥 Received input for name screening: {input_json}")

        if not input_json:
            logger.warning("❌ Empty or invalid JSON payload.")
            return jsonify({"error": "Empty or invalid JSON input"}), 400

        # Handle single dict or list of transactions
        if isinstance(input_json, dict):
            data_list = [input_json]
        elif isinstance(input_json, list) and all(isinstance(i, dict) for i in input_json):
            data_list = input_json
        else:
            logger.error("❌ Invalid data format received.")
            return jsonify({"error": "Input must be a JSON object or list of objects"}), 400

        df = pd.DataFrame(data_list)

        # Validate required columns
        required_fields = ["sender_name", "recipient_name", "transaction_id"]
        missing = [field for field in required_fields if field not in df.columns]
        if missing:
            logger.error(f"❌ Missing required fields: {missing}")
            return jsonify({"error": f"Missing required fields: {missing}"}), 400

        result = run_name_screening(df)
        logger.info("✅ Name screening completed successfully.")
        return jsonify(result), 200

    except Exception as e:
        logger.error(f"❌ Name screening failed: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

@app.route("/upload-aml-list", methods=["POST"])
def upload_aml_list():
    """
    API endpoint to upload and replace the AML watchlist.
    Accepts a CSV file via multipart/form-data and stores it in the aml_watchlist table.
    """
    try:
        logger.info("📥 Received AML list upload request.")

        # Extract file from request
        file = request.files.get("file")
        if not file:
            logger.warning("❌ No file part in the request.")
            return jsonify({"error": "No file uploaded"}), 400

        # Attempt to read the file into a DataFrame
        try:
            df = pd.read_csv(file)
        except Exception as e:
            logger.error(f"❌ Failed to read CSV: {e}")
            return jsonify({"error": f"Invalid CSV file: {e}"}), 400

        logger.info(f"📄 Uploaded file with {df.shape[0]} rows and {df.shape[1]} columns.")

        # Ensure required column exists
        if "name" not in df.columns:
            logger.warning("❌ Missing required 'name' column.")
            return jsonify({"error": "Missing required column: 'name'"}), 400

        # Normalize data
        df["name"] = df["name"].astype(str).str.strip().str.lower()
        df["status"] = "Active"

        # Add default values if not present
        df["risk_score"] = df.get("risk_score", 80)
        df["risk_category"] = df.get("risk_category", "Sanctions")
        df["entity_type"] = df.get("entity_type", "Individual")

        # Setup database engine
        engine = create_engine(
            f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
            f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
        )

        with engine.begin() as conn:
            logger.info("🧹 Truncating existing aml_watchlist table...")
            conn.execute(text("TRUNCATE TABLE aml_watchlist"))

            logger.info("📥 Inserting new AML records...")
            df.to_sql("aml_watchlist", con=conn, index=False, if_exists="append")

        logger.info("✅ AML list upload completed successfully.")
        return jsonify({"message": "AML list updated successfully"}), 200

    except Exception as e:
        logger.exception("❌ Unexpected error during AML upload.")
        return jsonify({"error": f"Failed to process file: {str(e)}"}), 500

@app.route("/transaction-screening", methods=["POST"])
def transaction_screening_route():
    try:
        input_json = request.get_json()
        logger.info(f"📥 Received transaction screening request: {input_json}")

        if not input_json:
            logger.warning("❌ Empty or invalid JSON input.")
            return jsonify({"error": "Empty or invalid JSON input"}), 400

        # Handle single dict or list of dicts
        if isinstance(input_json, dict):
            df = pd.DataFrame([input_json])  # single txn -> dataframe with 1 row
        elif isinstance(input_json, list) and all(isinstance(item, dict) for item in input_json):
            df = pd.DataFrame(input_json)  # multiple txns
        else:
            logger.error("❌ Input JSON must be an object or list of objects")
            return jsonify({"error": "Input JSON must be an object or list of objects"}), 400

        # Run your transaction screening rules, returns DataFrame with results
        results_df = run_transaction_rules(df)

        # Convert results DataFrame to list of dicts (JSON serializable)
        results = results_df.to_dict(orient="records")

        logger.info("✅ Transaction screening completed successfully.")
        return jsonify({"results": results}), 200

    except Exception as e:
        logger.error(f"❌ Error in /transaction-screening: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

@app.route('/explain/shap_outputs/<path:filename>')
def serve_shap_file(filename: str):
    return send_from_directory(SHAP_OUTPUT_DIR, filename)


@app.route('/reports/<path:filename>')
def serve_report_file(filename: str):
    return send_from_directory(REPORT_OUTPUT_DIR, filename)


# Added route to serve shap summary plot from /output folder
@app.route('/output/<path:filename>')
def serve_output_file(filename: str):
    return send_from_directory(OUTPUT_BASE_DIR, filename)


if __name__ == "__main__":
    app.run(debug=True)
