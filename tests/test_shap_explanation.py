import pandas as pd
from explain.explain_utils import load_components, preprocess_input, explain_transaction

# Load data
df = pd.read_csv("C:/Users/somas/PycharmProjects/FinGuardPro/data/sample_transactions.csv")

# Load model & encoders
model, scaler, encoders = load_components(
    "C:/Users/somas/PycharmProjects/FinGuardPro/models/fraud_detection_model.h5",
    "C:/Users/somas/PycharmProjects/FinGuardPro/models/scaler.pkl",
    "C:/Users/somas/PycharmProjects/FinGuardPro/models/label_encoders.pkl"
)

# Prepare input
X_scaled, feature_names = preprocess_input(df, scaler, encoders)

# Run SHAP explanation for first sample
explain_transaction(model, X_scaled, feature_names, idx=0)