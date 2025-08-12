import pandas as pd
import joblib
from sklearn.utils import resample
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import OrdinalEncoder

# === Load Model and Test Data ===
model = joblib.load("C:/Users/somas/PycharmProjects/FinGuardPro/models/xgboost_best_model.pkl")
X_test = pd.read_csv("C:/Users/somas/PycharmProjects/FinGuardPro/data/X_test.csv")
y_test = pd.read_csv("C:/Users/somas/PycharmProjects/FinGuardPro/data/y_test.csv")

# === Combine X and y for resampling ===
df_test = X_test.copy()
df_test['is_fraud'] = y_test.values

# === Categorical Encoding Function ===
def encode_categorical(df):
    df = df.copy()
    cat_cols = df.select_dtypes(include='object').columns
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    df[cat_cols] = encoder.fit_transform(df[cat_cols])
    return df

# === Evaluation Function ===
def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    auc = roc_auc_score(y, y_pred)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    return auc, precision, recall, f1

# === Simulate Different Fraud Rates ===
fraud_rates = [0.005, 0.01, 0.05]
results = []

for rate in fraud_rates:
    fraud = df_test[df_test['is_fraud'] == 1]
    non_fraud = df_test[df_test['is_fraud'] == 0]

    if len(fraud) == 0:
        raise ValueError("No fraud samples found!")

    n_fraud = len(fraud)
    n_non_fraud = int((n_fraud * (1 - rate)) / rate)
    n_non_fraud = min(n_non_fraud, len(non_fraud))  # limit to available samples

    non_fraud_downsampled = resample(non_fraud, replace=False, n_samples=n_non_fraud, random_state=42)

    df_sampled = pd.concat([fraud, non_fraud_downsampled]).sample(frac=1, random_state=42)

    X_sampled = df_sampled.drop(columns=['is_fraud'])
    y_sampled = df_sampled['is_fraud']

    X_encoded = encode_categorical(X_sampled)

    auc, precision, recall, f1 = evaluate_model(model, X_encoded, y_sampled)

    results.append({
        "Fraud Rate": f"{rate * 100:.1f}%",
        "AUC": round(auc, 4),
        "Precision": round(precision, 2),
        "Recall": round(recall, 2),
        "F1-Score": round(f1, 2)
    })

# === Display Results ===
results_df = pd.DataFrame(results)
print("\n📊 Model Robustness at Different Fraud Rates:")
print(results_df.to_string(index=False))
