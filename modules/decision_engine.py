import pandas as pd
import joblib
from tensorflow.keras.models import load_model

from modules.transaction_screening import run_transaction_screening
from modules.name_screening import run_name_screening
from utils.preprocessing import preprocess_for_model

# Load model and encoders once
MODEL_PATH = "C:/Users/somas/PycharmProjects/FinGuardPro/models/fraud_detection_model.h5"
SCALER_PATH = "C:/Users/somas/PycharmProjects/FinGuardPro/models/scaler.pkl"
ENCODERS_PATH = "C:/Users/somas/PycharmProjects/FinGuardPro/models/label_encoders.pkl"

model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
encoders = joblib.load(ENCODERS_PATH)

# Main function: Run complete risk detection
def run_decision_engine(transactions_df):
    # Step 1: Preprocess and score with ML model
    X = preprocess_for_model(transactions_df.copy(), scaler, encoders)
    fraud_scores = model.predict(X).flatten()
    transactions_df["fraud_score"] = fraud_scores

    # Step 2: Rule-based detection
    rules_df = run_transaction_screening(transactions_df)

    # Step 3: Name screening (AML match)
    aml_df = run_name_screening(transactions_df)

    # Step 4: Decision logic
    final_results = []
    for _, txn in transactions_df.iterrows():
        txn_id = txn["transaction_id"]
        risk_score = txn["fraud_score"]

        # Get rule violations (if any)
        rule_row = rules_df[rules_df["transaction_id"] == txn_id]
        rule_flags = rule_row["rule_violations"].values[0] if not rule_row.empty else []

        # Get AML match results
        aml_row = aml_df[aml_df["transaction_id"] == txn_id]
        if not aml_row.empty:
            aml_data = aml_row.iloc[0]
            sender_flag = aml_data["sender_flag"]
            receiver_flag = aml_data["receiver_flag"]
            sender_match = aml_data["sender_match"]
            receiver_match = aml_data["receiver_match"]
        else:
            sender_flag = receiver_flag = False
            sender_match = receiver_match = None

        # Final decision logic
        flagged = (
            risk_score >= 0.7 or
            bool(rule_flags) or
            sender_flag or
            receiver_flag
        )

        final_results.append({
            "transaction_id": txn_id,
            "sender_id": txn["sender_id"],
            "receiver_id": txn["receiver_id"],
            "amount": txn["amount"],
            "timestamp": txn["timestamp"],
            "fraud_score": round(float(risk_score), 4),
            "rule_violations": rule_flags,
            "sender_aml_match": sender_match,
            "receiver_aml_match": receiver_match,
            "flagged": flagged
        })

    return pd.DataFrame(final_results)
