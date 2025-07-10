import pandas as pd
import numpy as np
import os
import sys

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate time-based, amount-based, and behavior-based features
    for fraud detection modeling.
    """
    df = df.copy()

    # --- Timestamp Features ---
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_night'] = df['hour'].apply(lambda x: 1 if pd.notnull(x) and (x < 6 or x > 22) else 0)
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if pd.notnull(x) and x >= 5 else 0)

    # --- Amount Features ---
    df['log_amount'] = np.log1p(df['transaction_amount'])
    df['is_high_value'] = (df['transaction_amount'] > 10000).astype(int)

    # --- Sender-Receiver Features ---
    df['sender_receiver_combo'] = df['sender_id'].astype(str) + "_" + df['recipient_id'].astype(str)
    df['is_new_receiver'] = (~df.duplicated(subset=['sender_id', 'recipient_id'])).astype(int)

    # --- Sender Behavior Aggregates ---
    df['sender_txn_count'] = df.groupby('sender_id')['transaction_id'].transform('count')
    df['sender_avg_amt'] = df.groupby('sender_id')['transaction_amount'].transform('mean')
    df['amount_to_avg_ratio'] = df['transaction_amount'] / (df['sender_avg_amt'] + 1e-3)

    # --- Cleanup ---
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0, inplace=True)

    return df


def load_and_engineer(input_path: str, output_path: str) -> None:
    """
    Utility function to load raw transaction data,
    perform feature engineering, and save the processed dataset.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"❌ File not found: {input_path}")

    print("📂 Loading dataset...")
    df = pd.read_csv(input_path)
    print(f"✅ Loaded {len(df)} rows from {input_path}")

    # Optional renaming for compatibility
    rename_map = {
        'amount': 'transaction_amount'
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    # Validate required columns
    required_cols = ['transaction_id', 'timestamp', 'sender_id', 'recipient_id', 'transaction_amount']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"❌ Missing required columns: {missing}")

    print("🛠️ Engineering features...")
    df_engineered = engineer_features(df)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_engineered.to_csv(output_path, index=False)
    print(f"✅ Saved engineered dataset to {output_path}")


# Allow standalone execution as a script
if __name__ == "__main__":
    default_input = "C:/Users/somas/PycharmProjects/FinGuardPro/data/synthetic_dataset_large.csv"
    default_output = "C:/Users/somas/PycharmProjects/FinGuardPro/data/engineered_dataset_large.csv"

    input_path = sys.argv[1] if len(sys.argv) > 1 else default_input
    output_path = sys.argv[2] if len(sys.argv) > 2 else default_output

    load_and_engineer(input_path, output_path)
