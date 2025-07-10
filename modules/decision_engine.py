import pandas as pd

def combine_decision(
    transactions_df: pd.DataFrame,
    model_preds_df: pd.DataFrame,
    rule_df: pd.DataFrame,
    name_df: pd.DataFrame,
    threshold: float = 0.85
) -> pd.DataFrame:
    """
    Combines model predictions, rule-based logic, and name screening into final fraud decision.
    """
    df = transactions_df.copy()

    # Merge prediction, rules, name screening
    df = df.merge(model_preds_df, on="transaction_id", how="left")
    df = df.merge(rule_df, on="transaction_id", how="left")
    df = df.merge(name_df, on="transaction_id", how="left")

    # Fill missing flags with False
    flag_cols = [
        "sender_flag", "receiver_flag",
        "structuring_flag", "round_tripping_flag",
        "repeated_recipient_flag", "odd_hour_flag"
    ]
    for col in flag_cols:
        if col not in df.columns:
            df[col] = False
        df[col] = df[col].fillna(False).astype(bool)

    # Fill missing fraud scores
    df["fraud_score"] = df["fraud_score"].fillna(0.0)

    # Model-based fraud flag
    df["model_flag"] = df["fraud_score"] >= threshold

    # Rule-based fraud flag (if 2+ rules triggered)
    rule_columns = [
        "structuring_flag", "round_tripping_flag",
        "repeated_recipient_flag", "odd_hour_flag"
    ]
    df["rule_flag"] = df[rule_columns].sum(axis=1) >= 2

    # Final fraud decision
    df["final_flag"] = (
        df["model_flag"] |
        df["rule_flag"] |
        df["sender_flag"] |
        df["receiver_flag"]
    )

    # Generate explanation for each transaction
    df["reasons"] = df.apply(
        lambda row: get_reasons(row, threshold, rule_columns),
        axis=1
    )

    return df


def get_reasons(row: pd.Series, threshold: float, rule_columns: list) -> list:
    """Generate list of reasons for flagging a transaction."""
    reasons = []

    if row.get("sender_flag"):
        reasons.append("Sender matched AML watchlist")

    if row.get("receiver_flag"):
        reasons.append("Recipient matched AML watchlist")

    for rule in rule_columns:
        if row.get(rule):
            readable = rule.replace("_flag", "").replace("_", " ").title()
            reasons.append(f"Rule triggered: {readable}")

    if row.get("model_flag"):
        score = row.get("fraud_score", 0.0)
        reasons.append(f"ML fraud score {score:.2f} ≥ threshold {threshold}")

    return reasons or ["No significant risk factors"]
