import pandas as pd
import numpy as np

def preprocess_for_model(df, scaler, encoders):
    df = df.copy()

    # Drop non-feature columns
    drop_cols = ["transaction_id", "timestamp", "sender_name", "receiver_name"]
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

    # Apply label encoders to categorical columns
    for col, encoder in encoders.items():
        if col in df.columns:
            df[col] = df[col].map(lambda x: encoder.get(x, -1))  # Unknown values → -1

    # Fill any remaining NaNs with 0 (safe default)
    df.fillna(0, inplace=True)

    # Scale the features
    scaled = scaler.transform(df)

    return scaled
