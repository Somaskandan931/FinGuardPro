import pandas as pd
from typing import Dict, List, Any
from utils.config import feature_columns, EXCLUDE_FROM_MODEL

def safe_label_encode(series: pd.Series, le: Any) -> pd.Series:
    """
    Encode a pandas Series using the given label encoder, safely handling missing or unknown values.
    Unknown or null values are encoded as -1.
    """
    def encode_val(x):
        if pd.isnull(x):
            return -1
        try:
            # Check if value exists in label encoder classes before encoding
            return le.transform([x])[0] if x in le.classes_ else -1
        except Exception:
            return -1
    return series.apply(encode_val)


def preprocess_features(
    df: pd.DataFrame,
    label_encoders: Dict[str, Any],
    scaler: Any,
    categorical_columns: List[str]
) -> pd.DataFrame:
    """
    Preprocess the input dataframe by encoding categorical features,
    scaling numerical features, and ensuring correct column ordering.

    Args:
        df (pd.DataFrame): Input raw data frame.
        label_encoders (Dict[str, object]): Dictionary of fitted label encoders keyed by column.
        scaler (object): Fitted scaler object with transform method.
        categorical_columns (List[str]): List of categorical column names.

    Returns:
        pd.DataFrame: Preprocessed dataframe ready for model input.
    """
    df = df.copy()

    # Encode categorical columns safely, assign -1 if missing or unknown category
    for col in categorical_columns:
        if col in df.columns:
            le = label_encoders.get(col)
            if le is not None:
                df[col] = safe_label_encode(df[col], le)
            else:
                # If no encoder found, fill with -1
                df[col] = -1
        else:
            df[col] = -1  # default encoded value for missing categorical column

    # Drop columns excluded from the model
    df.drop(columns=EXCLUDE_FROM_MODEL, inplace=True, errors='ignore')

    # Add missing features with default value 0
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    # Reorder dataframe columns to match exactly the model's feature_columns order
    # Defensive: keep only columns present in df to avoid KeyErrors, then add missing as zeros
    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        for col in missing_cols:
            df[col] = 0

    df = df[feature_columns]

    # Identify numeric columns (categoricals are encoded as integers)
    numeric_cols = [col for col in feature_columns if col not in categorical_columns]

    # Scale only numeric columns
    df_numeric_scaled = pd.DataFrame(
        scaler.transform(df[numeric_cols]),
        columns=numeric_cols,
        index=df.index
    )

    # Combine scaled numeric columns with categorical columns (encoded integers)
    df_final = pd.concat([df_numeric_scaled, df[categorical_columns]], axis=1)

    # Ensure final columns exactly match feature_columns order
    df_final = df_final[feature_columns]

    return df_final
