import pandas as pd

def detect_structuring(transactions, amount_threshold=199999):
    return transactions[transactions['transaction_amount'] == amount_threshold]['transaction_id'].tolist()

def detect_round_tripping(transactions):
    seen_pairs = set()
    flagged = []
    for _, row in transactions.iterrows():
        pair = (row['sender_name'], row['recipient_name'])
        reverse = (row['recipient_name'], row['sender_name'])
        if reverse in seen_pairs:
            flagged.append(row['transaction_id'])
        seen_pairs.add(pair)
    return flagged

def detect_repeated_recipients(transactions, min_repeats=3):
    grouped = transactions.groupby(['sender_name', 'recipient_name']).size().reset_index(name='count')
    repeated = grouped[grouped['count'] >= min_repeats]
    repeated_pairs = repeated[['sender_name', 'recipient_name']].apply(tuple, axis=1).tolist()
    return transactions[transactions[['sender_name', 'recipient_name']].apply(tuple, axis=1).isin(repeated_pairs)]['transaction_id'].tolist()

def detect_odd_hours(transactions, start_hour=6, end_hour=23):
    if 'timestamp' not in transactions.columns:
        transactions['timestamp'] = pd.Timestamp.now()
    elif not pd.api.types.is_datetime64_any_dtype(transactions['timestamp']):
        transactions['timestamp'] = pd.to_datetime(transactions['timestamp'], errors='coerce').fillna(pd.Timestamp.now())
    return transactions[
        (transactions['timestamp'].dt.hour < start_hour) |
        (transactions['timestamp'].dt.hour > end_hour)
    ]['transaction_id'].tolist()

def run_rule_engine(transactions):
    structuring_ids = detect_structuring(transactions)
    round_trip_ids = detect_round_tripping(transactions)
    repeated_ids = detect_repeated_recipients(transactions)
    odd_hour_ids = detect_odd_hours(transactions)

    results = []
    for _, row in transactions.iterrows():
        tid = row['transaction_id']
        reasons = []

        if tid in structuring_ids:
            reasons.append("Structuring")
        if tid in round_trip_ids:
            reasons.append("Round-Tripping")
        if tid in repeated_ids:
            reasons.append("Repeated Recipient")
        if tid in odd_hour_ids:
            reasons.append("Odd Hour Activity")

        results.append({
            "transaction_id": tid,
            "rule_violations": reasons,
            "rule_flag": len(reasons) > 0
        })

    return pd.DataFrame(results)
