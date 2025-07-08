import json
import os

import pandas as pd

from backend.api_server import categorical_columns, label_encoders, safe_label_encode, BASE_MODEL_PATH, \
    EXCLUDE_FROM_MODEL, scaler

# Load saved feature columns once when starting the app
FEATURE_LIST_PATH = os.path.join(BASE_MODEL_PATH, "feature_columns.json")
with open(FEATURE_LIST_PATH, "r") as f:
    feature_columns = json.load(f)
print(f"✅ Loaded feature columns for scaler: {feature_columns}")

# Modify your preprocess_features function:

def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Encode categorical columns (your existing code)
    for col in categorical_columns:
        if col in df.columns:
            le = label_encoders[col]
            df[col + "_encoded"] = safe_label_encode(df[col], le)
        else:
            df[col + "_encoded"] = -1

    df.drop(columns=categorical_columns, inplace=True, errors='ignore')

    # Drop excluded columns not used for the model input
    df.drop(columns=EXCLUDE_FROM_MODEL, inplace=True, errors='ignore')

    # Add missing columns with zeros
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    # Reorder columns to exactly match the feature_columns order
    df = df[feature_columns]

    # Scale features using scaler (expects the exact columns and order)
    X_scaled = scaler.transform(df)

    return X_scaled
