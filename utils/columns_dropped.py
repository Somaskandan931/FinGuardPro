import pandas as pd

# Load your dataset as in preprocessing
df = pd.read_csv("C:/Users/somas/PycharmProjects/FinGuardPro/data/engineered_dataset_large.csv")

# Drop the columns you specified
drop_cols = [
    'transaction_id', 'timestamp',
    'sender_name', 'sender_upi_handle',
    'recipient_name', 'recipient_upi_handle',
    'fraud_type', 'sender_receiver_combo', 'sender_avg_amt'
]
df.drop(columns=drop_cols, inplace=True, errors='ignore')

# Identify categorical columns
categorical_cols = df.select_dtypes(include='object').columns.tolist()

# After label encoding, original categorical cols dropped, so features are:
# all numeric columns + *_encoded columns for each categorical column

# So create encoded column names
encoded_cols = [col + '_encoded' for col in categorical_cols]

# Now, drop original categorical columns
df.drop(columns=categorical_cols, inplace=True)

# Your features are:
feature_columns = list(df.drop(columns=['is_fraud']).columns) + encoded_cols

# Remove any duplicates just in case
feature_columns = list(set(feature_columns))

print(sorted(feature_columns))
