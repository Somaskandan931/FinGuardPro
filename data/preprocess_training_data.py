import pandas as pd
import numpy as np
import os
import pickle
import json
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

data_path = "//data/synthetic_dataset_large.csv"
df = pd.read_csv(data_path)

# Your ML model features — exclude non-features like timestamp, is_fraud, recipient_id
feature_columns = [
    'sender_balance_before', 'sender_age', 'recipient_balance_before',
    'transaction_type', 'device_type', 'location', 'merchant_category',
    'amount', 'hour_of_day', 'day_of_week', 'is_weekend',
    'txns_last_hour', 'txns_last_day', 'txns_last_week',
    'amount_to_balance_ratio', 'amount_vs_channel_limit_ratio',
    'is_round_amount', 'is_high_value', 'log_amount', 'is_new_receiver',
    'sender_txn_count', 'amount_to_avg_ratio',
    'sender_account_type', 'sender_risk_profile', 'recipient_account_type'
]

label_column = 'is_fraud'

# Categorical and numerical columns
categorical_cols = [
    'transaction_type', 'device_type', 'location', 'merchant_category',
    'sender_account_type', 'sender_risk_profile', 'recipient_account_type'
]
numerical_cols = [col for col in feature_columns if col not in categorical_cols]

# Fill missing categorical data
for col in categorical_cols:
    df[col] = df[col].fillna('UNKNOWN').astype(str)

# Fit label encoders on categorical columns
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Fill missing numerical data with median
for col in numerical_cols:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].median())

# Scale numerical features
scaler = MinMaxScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Prepare X and y
X = df[feature_columns]
y = df[label_column]

# Split train/test stratified by fraud label
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Save encoders, scaler, and feature columns for inference use
output_dir = "//data"
os.makedirs(output_dir, exist_ok=True)

with open(os.path.join(output_dir, 'label_encoders.pkl'), 'wb') as f:
    pickle.dump(encoders, f)

with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)

with open(os.path.join(output_dir, 'feature_columns.json'), 'w') as f:
    json.dump(feature_columns, f)

# Save train/test splits
X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)

print("✅ Preprocessing complete and saved.")
