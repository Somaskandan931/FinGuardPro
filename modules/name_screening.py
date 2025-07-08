import pandas as pd
from rapidfuzz import fuzz, process
from pathlib import Path

WATCHLIST_FILE = Path("data/aml_watchlist.csv")

def load_watchlist(filepath=WATCHLIST_FILE):
    if not filepath.exists():
        return pd.DataFrame(columns=["name"])
    df = pd.read_csv(filepath)
    df["name"] = df["name"].astype(str).str.lower().str.strip()
    return df.drop_duplicates()

def check_name_match(name, watchlist_df, threshold=85):
    if watchlist_df.empty or not isinstance(name, str) or not name.strip():
        return None, 0
    name = name.lower().strip()
    match = process.extractOne(name, watchlist_df["name"], scorer=fuzz.token_sort_ratio)
    if match and match[1] >= threshold:
        return match[0], match[1]
    return None, 0

def run_name_screening(transactions_df, threshold=85):
    watchlist_df = load_watchlist()
    results = []

    for _, row in transactions_df.iterrows():
        sender = str(row.get('sender_name', '')).strip()
        receiver = str(row.get('receiver_name', '')).strip()
        sender_match, sender_score = check_name_match(sender, watchlist_df, threshold)
        receiver_match, receiver_score = check_name_match(receiver, watchlist_df, threshold)

        results.append({
            "transaction_id": row.get("transaction_id"),
            "sender_flag": bool(sender_match),
            "sender_score": sender_score,
            "receiver_flag": bool(receiver_match),
            "receiver_score": receiver_score
        })

    return pd.DataFrame(results)
