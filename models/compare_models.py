import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

models = {
    "XGBoost" : "C:/Users/somas/PycharmProjects/FinGuardPro/models/xgboost_best_model.pkl",
    "LightGBM" : "C:/Users/somas/PycharmProjects/FinGuardPro/models/lightgbm_best_model.pkl",
    "CatBoost" : "C:/Users/somas/PycharmProjects/FinGuardPro/models/catboost_best_model.pkl"
}

# Load raw test data
X_test_raw = pd.read_csv( "C:/Users/somas/PycharmProjects/FinGuardPro/data/X_test.csv" )
y_test = pd.read_csv( "C:/Users/somas/PycharmProjects/FinGuardPro/data/y_test.csv" ).values.ravel()

# Replace this list with the exact features used during training (31 features)
training_features = ["sender_balance_before", "sender_age", "recipient_balance_before", "transaction_type", "device_type", "location", "merchant_category", "amount", "hour_of_day", "day_of_week", "is_weekend", "txns_last_hour", "txns_last_day", "txns_last_week", "amount_to_balance_ratio", "amount_vs_channel_limit_ratio", "is_round_amount", "is_high_value", "log_amount", "is_new_receiver", "sender_txn_count", "amount_to_avg_ratio", "sender_account_type", "sender_risk_profile", "recipient_account_type"]

# Fix X_test columns to match training
for col in training_features :
    if col not in X_test_raw.columns :
        X_test_raw[col] = 0  # or np.nan if you want, but 0 is safer for model input

X_test_aligned = X_test_raw[training_features]

results = []

for name, path in models.items() :
    print( f"\n=== {name} ===" )
    try :
        model = joblib.load( path )

        # For CatBoost, the input can be DataFrame, for others numpy array is safe
        X_input = X_test_aligned

        y_pred = model.predict( X_input )
        if hasattr( model, "predict_proba" ) :
            y_proba = model.predict_proba( X_input )[:, 1]
        else :
            y_proba = y_pred

        auc = roc_auc_score( y_test, y_proba )
        print( f"AUC: {auc:.4f}" )
        print( classification_report( y_test, y_pred ) )
        print( f"Confusion Matrix:\n{confusion_matrix( y_test, y_pred )}" )

        from sklearn.metrics import precision_score, recall_score, f1_score

        results.append( {
            "Model" : name,
            "Precision" : precision_score( y_test, y_pred ),
            "Recall" : recall_score( y_test, y_pred ),
            "F1-Score" : f1_score( y_test, y_pred ),
            "AUC" : auc
        } )
    except Exception as e :
        print( f"Error evaluating {name}: {e}" )

if results :
    print( "\nSummary Comparison:" )
    print( pd.DataFrame( results ).set_index( "Model" ) )
