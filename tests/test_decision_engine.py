import pandas as pd
from modules.decision_engine import run_decision_engine

def test_decision_pipeline():
    df = pd.DataFrame([{
        "transaction_id": "TXN001",
        "sender_id": "U123",
        "receiver_id": "U456",
        "amount": 199999,
        "timestamp": pd.Timestamp("2025-07-06 02:30:00"),
        "sender_name": "John Doe",
        "receiver_name": "Jane Smith"
    }])

    results = run_decision_engine(df)
    assert "fraud_score" in results.columns
    assert "flagged" in results.columns
    print("✅ Decision engine test passed")
    print(results)

if __name__ == "__main__":
    test_decision_pipeline()
