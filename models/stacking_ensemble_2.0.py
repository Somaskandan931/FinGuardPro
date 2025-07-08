import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

# ✅ Load preprocessed training and test data
X_train = pd.read_csv("C:/Users/somas/PycharmProjects/FinGuardPro/data/X_train.csv")
y_train = pd.read_csv("C:/Users/somas/PycharmProjects/FinGuardPro/data/y_train.csv").values.ravel()
X_test = pd.read_csv("C:/Users/somas/PycharmProjects/FinGuardPro/data/X_test.csv")
y_test = pd.read_csv("C:/Users/somas/PycharmProjects/FinGuardPro/data/y_test.csv").values.ravel()

# ✅ Load tuned base models
xgb_model = joblib.load("C:/Users/somas/PycharmProjects/FinGuardPro/models/xgboost_tuned_model.pkl")
lgbm_model = joblib.load("C:/Users/somas/PycharmProjects/FinGuardPro/models/lgbm_tuned_model.pkl")
catboost_model = joblib.load("C:/Users/somas/PycharmProjects/FinGuardPro/models/catboost_tuned_model.pkl")

# ✅ Generate meta-features from base model predictions (training set)
train_meta_features = np.column_stack([
    xgb_model.predict_proba(X_train)[:, 1],
    lgbm_model.predict_proba(X_train)[:, 1],
    catboost_model.predict_proba(X_train)[:, 1]
])

# ✅ Train the stacking meta-model (Logistic Regression)
meta_model = LogisticRegression(max_iter=1000, solver='liblinear')
meta_model.fit(train_meta_features, y_train)

# ✅ Save the meta-model
joblib.dump(meta_model, "C:/Users/somas/PycharmProjects/FinGuardPro/models/stacked_meta_model.pkl")
print("✅ Meta-model saved as 'stacked_meta_model.pkl'")

# ✅ Generate meta-features from base model predictions (test set)
test_meta_features = np.column_stack([
    xgb_model.predict_proba(X_test)[:, 1],
    lgbm_model.predict_proba(X_test)[:, 1],
    catboost_model.predict_proba(X_test)[:, 1]
])

# ✅ Predict and evaluate
y_pred = meta_model.predict(test_meta_features)
y_proba = meta_model.predict_proba(test_meta_features)[:, 1]

print("🔍 ROC AUC Score:", roc_auc_score(y_test, y_proba))
print("📊 Classification Report:")
print(classification_report(y_test, y_pred))
