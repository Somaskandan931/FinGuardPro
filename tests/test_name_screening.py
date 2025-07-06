import pandas as pd
from modules.name_screening import run_name_screening, add_names_to_watchlist

def test_aml_screening():
    # Add known suspicious name
    add_names_to_watchlist(["John Doe"])

    test_data = pd.DataFrame([{
        "transaction_id": "TXN001",
        "sender_name": "Jon Doe",
        "receiver_name": "Jane Smith"
    }])

    result = run_name_screening(test_data)
    assert result.iloc[0]["sender_flag"] is True
    print("✅ Name screening test passed")

if __name__ == "__main__":
    test_aml_screening()
