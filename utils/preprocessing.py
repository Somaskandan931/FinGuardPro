# preprocess_data.py

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Paths
INPUT_PATH = "C:/Users/somas/PycharmProjects/FinGuardPro/data/engineered_dataset_large.csv"
OUTPUT_DIR = "C:/Users/somas/PycharmProjects/FinGuardPro/data/"
MODEL_DIR = "C:/Users/somas/PycharmProjects/FinGuardPro/models/"

# Load engineered data
df = pd.read_csv(INPUT_PATH)
print(f"✅ Loaded engineered dataset: {len(df)} records")
print(f"💡 Fraud rate: {df['is_fraud'].mean():.2%}")

# Drop unnecessary columns
drop_cols = [
    'transaction_id', 'timestamp',
    'sender_name', 'sender_upi_handle',
    'recipient_name', 'recipient_upi_handle',
    'fraud_type', 'sender_receiver_combo', 'sender_avg_amt'
]
df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True, errors='ignore')
print(f"🧹 Dropped non-feature columns")

# Label encode object columns
categorical_cols = df.select_dtypes(include='object').columns.tolist()
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le
    print(f"✔️ Encoded: {col}")

# Save label encoders
with open(MODEL_DIR + "label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)
print("💾 Label encoders saved.")

# Drop original object columns
df.drop(columns=categorical_cols, inplace=True)
print("🧼 Dropped original categorical columns")

# Remove problematic columns
invalid_cols = []
for col in df.drop(columns=["is_fraud"]).columns:
    if df[col].isna().all() or df[col].nunique() <= 1 or np.isinf(df[col]).any():
        invalid_cols.append(col)

df.drop(columns=invalid_cols, inplace=True)
df.fillna(0, inplace=True)

# Scale numerical features
scaler = StandardScaler()
features = df.drop(columns=["is_fraud"]).columns
df[features] = scaler.fit_transform(df[features])

# Save scaler
with open(MODEL_DIR + "scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("💾 Scaler saved.")

# Split into X and y
X = df.drop("is_fraud", axis=1)
y = df["is_fraud"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Save splits
X_train.to_csv(OUTPUT_DIR + "X_train.csv", index=False)
X_test.to_csv(OUTPUT_DIR + "X_test.csv", index=False)
y_train.to_csv(OUTPUT_DIR + "y_train.csv", index=False)
y_test.to_csv(OUTPUT_DIR + "y_test.csv", index=False)
print("📦 Train-test splits saved.")
