import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load saved base models
xgb_model = joblib.load("C:/Users/somas/PycharmProjects/FinGuardPro/models/xgboost_best_model.pkl")
lgbm_model = joblib.load("C:/Users/somas/PycharmProjects/FinGuardPro/models/lightgbm_best_model.pkl")
catboost_model = joblib.load("C:/Users/somas/PycharmProjects/FinGuardPro/models/catboost_best_model.pkl")

# Load train and test data
X_train = pd.read_csv("C:/Users/somas/PycharmProjects/FinGuardPro/data/X_train.csv")
y_train = pd.read_csv("C:/Users/somas/PycharmProjects/FinGuardPro/data/y_train.csv").values.ravel()
X_test = pd.read_csv("C:/Users/somas/PycharmProjects/FinGuardPro/data/X_test.csv")
y_test = pd.read_csv("C:/Users/somas/PycharmProjects/FinGuardPro/data/y_test.csv").values.ravel()

# Get base model predictions (probabilities) on training set for meta training
train_meta_features = np.column_stack([
    xgb_model.predict_proba(X_train)[:, 1],
    lgbm_model.predict_proba(X_train)[:, 1],
    catboost_model.predict_proba(X_train)[:, 1]
])

# Train meta-model on base model predictions
meta_model = LogisticRegression()
meta_model.fit(train_meta_features, y_train)

# Save meta-model
joblib.dump(meta_model, "C:/Users/somas/PycharmProjects/FinGuardPro/models/meta_model.pkl")

# Prepare meta features on test set
test_meta_features = np.column_stack([
    xgb_model.predict_proba(X_test)[:, 1],
    lgbm_model.predict_proba(X_test)[:, 1],
    catboost_model.predict_proba(X_test)[:, 1]
])

# Predict with meta-model
y_pred = meta_model.predict(test_meta_features)
y_proba = meta_model.predict_proba(test_meta_features)[:, 1]

# Evaluate
print(classification_report(y_test, y_pred))
