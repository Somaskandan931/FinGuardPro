import optuna
from catboost import CatBoostClassifier, Pool
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np
import joblib

def objective(trial):
    # Load data
    X = pd.read_csv( "//data/X_train.csv" )
    y = pd.read_csv( "//data/y_train.csv" ).values.ravel()

    # Categorical features (if any)
    cat_features = []  # Update this if you have categorical features

    # Suggest hyperparameters
    params = {
        'iterations': 1000,
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'random_strength': trial.suggest_float('random_strength', 0.0, 10.0),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'scale_pos_weight': (y == 0).sum() / (y == 1).sum(),
        'verbose': False,
        'eval_metric': 'AUC',
        'random_seed': 42,
        'early_stopping_rounds': 50
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        train_pool = Pool(X_train, y_train, cat_features=cat_features)
        val_pool = Pool(X_val, y_val, cat_features=cat_features)

        model = CatBoostClassifier(**params)
        model.fit(train_pool, eval_set=val_pool, use_best_model=True)

        preds = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, preds)
        aucs.append(auc)

    return np.mean(aucs)

if __name__ == "__main__":
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50, show_progress_bar=True)

    print("Best AUC:", study.best_value)
    print("Best params:", study.best_params)

    # 🔁 Retrain final model on full training data with best params
    print("Training final model on full data...")

    X = pd.read_csv( "//data/X_train.csv" )
    y = pd.read_csv( "//data/y_train.csv" ).values.ravel()
    cat_features = []  # Same as above

    best_params = study.best_params
    best_params.update({
        'iterations': 1000,
        'scale_pos_weight': (y == 0).sum() / (y == 1).sum(),
        'verbose': False,
        'eval_metric': 'AUC',
        'random_seed': 42
    })

    final_model = CatBoostClassifier(**best_params)
    final_model.fit(Pool(X, y, cat_features=cat_features))

    # 💾 Save the final model
    joblib.dump( final_model, "//models/catboost_tuned_model.pkl" )
    print("✅ Final tuned CatBoost model saved as 'catboost_tuned_model.pkl'")
