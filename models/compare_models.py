# compare_models.py

import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

models = {
    "XGBoost": "C:/Users/somas/PycharmProjects/FinGuardPro/models/xgboost_best_model.pkl",
    "LightGBM": "C:/Users/somas/PycharmProjects/FinGuardPro/models/lightgbm_best_model.pkl",
    "CatBoost": "C:/Users/somas/PycharmProjects/FinGuardPro/models/catboost_best_model.pkl"
}

X_test = pd.read_csv("C:/Users/somas/PycharmProjects/FinGuardPro/data/X_test.csv")
y_test = pd.read_csv("C:/Users/somas/PycharmProjects/FinGuardPro/data/y_test.csv").values.ravel()

for name, path in models.items():
    model = joblib.load(path)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_proba)
    print(f"\n=== {name} ===")
    print(f"AUC: {auc:.4f}")
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:\n{cm}")
