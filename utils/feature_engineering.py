import pandas as pd
import numpy as np

def engineer_features(df):
    df = df.copy()

    # Ensure timestamp is datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # --- Time-based Features ---
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_night'] = df['hour'].apply(lambda x: 1 if (x < 6 or x > 22) else 0)

    # --- Amount-based Features ---
    df['log_amount'] = np.log1p(df['transaction_amount'])
    df['is_high_value'] = (df['transaction_amount'] > 10000).astype(int)

    # --- Sender-Receiver Behavioral Features ---
    df['sender_receiver_combo'] = df['sender_id'].astype(str) + "_" + df['recipient_id'].astype(str)
    df['is_new_receiver'] = ~df.duplicated(subset=['sender_id', 'recipient_id']).astype(int)

    # --- Sender Activity Features ---
    df['sender_txn_count'] = df.groupby('sender_id')['transaction_id'].transform('count')
    df['sender_avg_amt'] = df.groupby('sender_id')['transaction_amount'].transform('mean')
    df['amount_to_avg_ratio'] = df['transaction_amount'] / (df['sender_avg_amt'] + 1e-3)

    # Replace infinities and fill NaNs
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0, inplace=True)

    return df

if __name__ == "__main__":
    INPUT_PATH = "C:/Users/somas/PycharmProjects/FinGuardPro/data/synthetic_dataset_large.csv"
    OUTPUT_PATH = "C:/Users/somas/PycharmProjects/FinGuardPro/data/engineered_dataset_large.csv"

    print("📂 Loading raw dataset...")
    df = pd.read_csv(INPUT_PATH)
    print(f"✅ Loaded {len(df)} rows")
    print(f"📄 Columns in file: {df.columns.tolist()}")

    # --- Rename columns for consistency ---
    rename_map = {
        'amount': 'transaction_amount'
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    # --- Validate required columns ---
    required_cols = ['transaction_id', 'timestamp', 'sender_id', 'recipient_id', 'transaction_amount']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"❌ Missing required columns: {missing}")

    print("🛠️ Engineering features...")
    df_engineered = engineer_features(df)

    print(f"💾 Saving engineered dataset to {OUTPUT_PATH}")
    df_engineered.to_csv(OUTPUT_PATH, index=False)
    print("✅ Feature engineering complete.")
