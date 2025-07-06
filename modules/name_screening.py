import pandas as pd
from rapidfuzz import fuzz, process
from pathlib import Path

WATCHLIST_FILE = Path("data/aml_watchlist.csv")

# Load AML watchlist
def load_watchlist(filepath=WATCHLIST_FILE):
    if not filepath.exists():
        return pd.DataFrame(columns=["name"])
    return pd.read_csv(filepath)

# Add names to watchlist
def add_names_to_watchlist(new_names, filepath=WATCHLIST_FILE):
    current = load_watchlist(filepath)
    new_df = pd.DataFrame({"name": new_names})
    combined = pd.concat([current, new_df], ignore_index=True).drop_duplicates()
    combined.to_csv(filepath, index=False)

# Match one name
def check_name_match(name, watchlist_df, threshold=85):
    if watchlist_df.empty or not name:
        return None, 0
    match = process.extractOne(name, watchlist_df["name"], scorer=fuzz.token_sort_ratio)
    if match and match[1] >= threshold:
        return match[0], match[1]
    return None, 0

# Run AML screening on a batch
def run_name_screening(transactions_df, filepath=WATCHLIST_FILE, threshold=85):
    watchlist_df = load_watchlist(filepath)
    results = []

    for _, row in transactions_df.iterrows():
        sender_name = row.get('sender_name', '')
        receiver_name = row.get('receiver_name', '')

        sender_match, sender_score = check_name_match(sender_name, watchlist_df, threshold)
        receiver_match, receiver_score = check_name_match(receiver_name, watchlist_df, threshold)

        results.append({
            "transaction_id": row.get("transaction_id"),
            "sender_name": sender_name,
            "receiver_name": receiver_name,
            "sender_flag": bool(sender_match),
            "sender_match": sender_match,
            "sender_score": sender_score,
            "receiver_flag": bool(receiver_match),
            "receiver_match": receiver_match,
            "receiver_score": receiver_score
        })

    return pd.DataFrame(results)
