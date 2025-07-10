import pandas as pd

def detect_structuring(transactions: pd.DataFrame, amount_threshold: float = 199999) -> list:
    """Detect structuring transactions (suspicious amount exactly at threshold)."""
    if 'amount' not in transactions.columns:
        return []
    return transactions[transactions['amount'] == amount_threshold]['transaction_id'].tolist()

def detect_round_tripping(transactions: pd.DataFrame) -> list:
    """Detect round-tripping transactions where sender and receiver roles reverse."""
    if not {'sender_name', 'recipient_name', 'transaction_id'}.issubset(transactions.columns):
        return []

    seen_pairs = set()
    flagged = []

    for _, row in transactions.iterrows():
        pair = (row['sender_name'], row['recipient_name'])
        reverse = (row['recipient_name'], row['sender_name'])
        if reverse in seen_pairs:
            flagged.append(row['transaction_id'])
        seen_pairs.add(pair)

    return flagged

def detect_repeated_recipients(transactions: pd.DataFrame, min_repeats: int = 3) -> list:
    """Detect repeated transactions to same recipient from same sender."""
    required_cols = {'sender_name', 'recipient_name', 'transaction_id'}
    if not required_cols.issubset(transactions.columns):
        return []

    grouped = transactions.groupby(['sender_name', 'recipient_name']).size().reset_index(name='count')
    repeated = grouped[grouped['count'] >= min_repeats]
    repeated_pairs = set(repeated[['sender_name', 'recipient_name']].apply(tuple, axis=1))

    return transactions[
        transactions[['sender_name', 'recipient_name']].apply(tuple, axis=1).isin(repeated_pairs)
    ]['transaction_id'].tolist()

def detect_odd_hours(transactions: pd.DataFrame, start_hour: int = 6, end_hour: int = 23) -> list:
    """Detect transactions occurring during suspicious hours."""
    if 'timestamp' not in transactions.columns:
        return []

    if not pd.api.types.is_datetime64_any_dtype(transactions['timestamp']):
        transactions['timestamp'] = pd.to_datetime(transactions['timestamp'], errors='coerce')
    transactions['timestamp'] = transactions['timestamp'].fillna(pd.Timestamp.now())

    return transactions[
        (transactions['timestamp'].dt.hour < start_hour) |
        (transactions['timestamp'].dt.hour > end_hour)
    ]['transaction_id'].tolist()

def run_transaction_rules(transactions: pd.DataFrame) -> pd.DataFrame:
    """Run all rule-based checks and return flagged transaction info."""
    # Run individual rule checks
    structuring_ids = detect_structuring(transactions)
    round_trip_ids = detect_round_tripping(transactions)
    repeated_ids = detect_repeated_recipients(transactions)
    odd_hour_ids = detect_odd_hours(transactions)

    results = []

    for _, row in transactions.iterrows():
        tid = row.get('transaction_id')
        if not tid:
            continue

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
            "rule_flag": bool(reasons)
        })

    return pd.DataFrame(results)
