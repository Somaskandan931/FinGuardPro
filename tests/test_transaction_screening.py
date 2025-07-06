import pandas as pd
from modules.transaction_screening import run_transaction_screening

def test_rules():
    df = pd.DataFrame([
        {
            "transaction_id": "TXN001",
            "sender_id": "U123",
            "receiver_id": "U456",
            "amount": 199999,
            "timestamp": pd.Timestamp("2025-07-06 02:30:00")
        },
        {
            "transaction_id": "TXN002",
            "sender_id": "U456",
            "receiver_id": "U123",
            "amount": 1000,
            "timestamp": pd.Timestamp("2025-07-06 03:00:00")
        },
        {
            "transaction_id": "TXN003",
            "sender_id": "U123",
            "receiver_id": "U456",
            "amount": 1000,
            "timestamp": pd.Timestamp("2025-07-06 03:00:00")
        },
        {
            "transaction_id": "TXN004",
            "sender_id": "U123",
            "receiver_id": "U456",
            "amount": 1000,
            "timestamp": pd.Timestamp("2025-07-06 03:00:00")
        }
    ])

    result = run_transaction_screening(df)
    assert not result.empty
    print("✅ Transaction screening test passed")
    print(result)

if __name__ == "__main__":
    test_rules()
