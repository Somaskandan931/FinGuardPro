import pandas as pd

def detect_structuring(transactions, amount_threshold=199999):
    # Transactions exactly at the threshold amount indicate possible structuring
    return transactions[transactions['amount'] == amount_threshold]['transaction_id'].tolist()

def detect_round_tripping(transactions):
    seen_pairs = set()
    flagged = []
    for _, row in transactions.iterrows():
        pair = (row['sender_id'], row['receiver_id'])
        reverse = (row['receiver_id'], row['sender_id'])
        if reverse in seen_pairs:
            flagged.append(row['transaction_id'])
        seen_pairs.add(pair)
    return flagged

def detect_repeated_recipients(transactions, min_repeats=3):
    grouped = transactions.groupby(['sender_id', 'receiver_id']).size().reset_index(name='count')
    repeated = grouped[grouped['count'] >= min_repeats]
    repeated_pairs = repeated[['sender_id', 'receiver_id']].apply(tuple, axis=1).tolist()
    # Return all transaction_ids that belong to repeated pairs
    return transactions[transactions[['sender_id', 'receiver_id']].apply(tuple, axis=1).isin(repeated_pairs)]['transaction_id'].tolist()

def detect_odd_hours(transactions, start_hour=6, end_hour=23):
    # Ensure timestamp is datetime type
    if not pd.api.types.is_datetime64_any_dtype(transactions['timestamp']):
        transactions['timestamp'] = pd.to_datetime(transactions['timestamp'], errors='coerce')
    return transactions[
        (transactions['timestamp'].dt.hour < start_hour) |
        (transactions['timestamp'].dt.hour > end_hour)
    ]['transaction_id'].tolist()

def run_rule_engine(transactions):
    """
    Apply multiple rule-based detections on the transaction DataFrame.
    Returns a DataFrame with transaction_ids and list of violated rules.
    """

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

        if reasons:
            results.append({
                "transaction_id": tid,
                "rule_violations": reasons,
                "rule_flag": True
            })
        else:
            results.append({
                "transaction_id": tid,
                "rule_violations": [],
                "rule_flag": False
            })

    return pd.DataFrame(results)
