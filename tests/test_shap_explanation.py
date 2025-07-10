import pandas as pd
from utils.explain_utils import (
    load_model_and_preprocessor,
    preprocess_input,
    explain_transaction,
    explain_summary_plot
)

if __name__ == "__main__":
    # STEP 1: Load model, scaler, encoders
    model, scaler, encoders = load_model_and_preprocessor()

    # STEP 2: Load sample test transactions
    test_csv_path = "//data/sample_transactions_full.csv"
    raw_df = pd.read_csv(test_csv_path)

    # STEP 3: Preprocess the data
    input_df = preprocess_input(raw_df, scaler, encoders)

    # STEP 4: Get transaction IDs for filenames
    transaction_ids = raw_df["transaction_id"].tolist() if "transaction_id" in raw_df.columns else None

    # STEP 5: Generate individual SHAP explanations
    explain_transaction(
        model=model,
        input_df=input_df,
        transaction_ids=transaction_ids,
        top_n=10
    )

    # STEP 6: Generate SHAP summary plot
    explain_summary_plot(
        model=model,
        input_df=input_df
    )
