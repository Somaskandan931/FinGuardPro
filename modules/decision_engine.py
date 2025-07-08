import pandas as pd


def combine_decision ( transactions_df, model_preds_df, rule_df, name_df, threshold=0.5 ) :
    """
    Combines model score, rule results, and name screening into final decision.
    Returns: DataFrame with final fraud flags.
    """
    df = transactions_df.copy()
    df = df.merge( model_preds_df, on="transaction_id", how="left" )
    df = df.merge( rule_df, on="transaction_id", how="left" )
    df = df.merge( name_df, on="transaction_id", how="left" )

    df["model_flag"] = df["fraud_score"] >= threshold
    df["final_flag"] = (
            df["model_flag"] |
            df["rule_flag"] |
            df["sender_flag"] |
            df["receiver_flag"]
    )

    return df
