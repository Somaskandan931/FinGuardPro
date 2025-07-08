from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import traceback
import pickle
import joblib
import os

from modules.rule_engine import run_rule_engine
from modules.name_screening import run_name_screening
from modules.decision_engine import combine_decision
from reports.report_utils import generate_fraud_report
from explain.explain_utils import explain_transaction, explain_summary_plot

app = Flask(__name__)

# Load model components once at startup
model = joblib.load("C:/Users/somas/PycharmProjects/FinGuardPro/models/stacked_meta_model.pkl")
with open("C:/Users/somas/PycharmProjects/FinGuardPro/models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("C:/Users/somas/PycharmProjects/FinGuardPro/models/label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

categorical_columns = list(label_encoders.keys())

feature_columns = [
    'amount_to_avg_ratio', 'amount_to_balance_ratio', 'amount_vs_channel_limit_ratio',
    'day_of_week', 'device_type_encoded', 'hour', 'hour_of_day', 'is_high_value',
    'is_impossible_travel', 'is_new_receiver', 'is_night', 'is_round_amount',
    'is_round_trip', 'is_weekend', 'location_encoded', 'log_amount',
    'merchant_category_encoded', 'recipient_account_type_encoded', 'recipient_balance_before',
    'recipient_id_encoded', 'recipient_suspicious_name', 'round_trip_chain_id',
    'round_trip_position', 'sender_account_type_encoded', 'sender_age',
    'sender_balance_before', 'sender_id_encoded', 'sender_risk_profile_encoded',
    'sender_suspicious_name', 'sender_txn_count', 'transaction_amount',
    'transaction_type_encoded', 'txns_last_day', 'txns_last_hour', 'txns_last_week'
]

def preprocess_features(df):
    df = df.copy()
    for col in categorical_columns:
        if col in df.columns:
            le = label_encoders[col]
            # Map unseen labels to -1 safely
            df[col + "_encoded"] = df[col].apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )
        else:
            df[col + "_encoded"] = -1
    df.drop(columns=categorical_columns, inplace=True, errors='ignore')
    X = df[feature_columns]
    return scaler.transform(X)

def determine_risk_level(score):
    if score >= 0.8:
        return "High"
    elif score >= 0.5:
        return "Medium"
    else:
        return "Low"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_json = request.get_json()
        if not input_json:
            return jsonify({"error": "Empty or invalid input."}), 400

        transactions_df = pd.DataFrame([input_json]) if isinstance(input_json, dict) else pd.DataFrame(input_json)

        # Run detection engines
        rules = run_rule_engine(transactions_df)
        names = run_name_screening(transactions_df)

        # Model predictions
        try:
            X = preprocess_features(transactions_df)
            preds = model.predict_proba(X)[:, 1]
        except Exception as e:
            print(f"❌ Model prediction error: {e}")
            preds = [0.0] * len(transactions_df)

        model_preds = pd.DataFrame({
            "transaction_id": transactions_df["transaction_id"],
            "fraud_score": preds
        })

        final_df = combine_decision(transactions_df, model_preds, rules, names)

        responses = []
        for idx, row in final_df.iterrows():
            txn_data = transactions_df.iloc[idx].to_dict()
            fraud_score = row.get("fraud_score", 0.0)
            risk_level = determine_risk_level(fraud_score)

            # Generate PDF report
            report_path = generate_fraud_report(
                txn_data=txn_data,
                fraud_score=fraud_score,
                risk_level=risk_level
            )

            # Generate SHAP waterfall plot for this transaction
            shap_paths = explain_transaction(
                model=model,
                input_df=transactions_df.iloc[[idx]],
                scaler=scaler,
                encoders=label_encoders,
                feature_columns=feature_columns,
                transaction_ids=[txn_data.get("transaction_id", f"tx_{idx}")],
                output_dir="explain/shap_outputs"
            )

            row_dict = row.to_dict()
            row_dict["risk_level"] = risk_level

            # Return relative paths for frontends to access via static routes
            row_dict["report_path"] = f"reports/{os.path.basename(report_path)}" if report_path else None
            row_dict["shap_plot"] = f"explain/shap_outputs/{os.path.basename(shap_paths[0])}" if shap_paths else None
            responses.append(row_dict)

        # If batch input, also generate SHAP summary plot
        if len(transactions_df) > 1:
            summary_path = explain_summary_plot(
                model=model,
                input_df=transactions_df,
                scaler=scaler,
                encoders=label_encoders,
                feature_columns=feature_columns,
                output_path="explain/shap_summary.png"
            )
            return jsonify({
                "results": responses,
                "shap_summary_plot": "explain/shap_summary.png"
            })

        return jsonify(responses[0] if len(responses) == 1 else responses)

    except Exception as e:
        print("❌ Error in /predict:", traceback.format_exc())
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

# Static file serving for SHAP images and reports
@app.route('/explain/shap_outputs/<path:filename>')
def serve_shap_file(filename):
    return send_from_directory('explain/shap_outputs', filename)

@app.route('/reports/<path:filename>')
def serve_report_file(filename):
    return send_from_directory('reports', filename)

if __name__ == "__main__":
    os.makedirs("explain/shap_outputs", exist_ok=True)
    os.makedirs("reports", exist_ok=True)  # Make sure report folder exists
    app.run(debug=True)
