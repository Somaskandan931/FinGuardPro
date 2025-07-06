import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, RepeatVector, TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.metrics import Precision, Recall
import joblib
import warnings
import os
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Check if GPU is available
print("GPU Available: ", tf.config.list_physical_devices('GPU'))
print("TensorFlow Version: ", tf.__version__)

# ===== FRAUD DETECTION MODEL CLASS =====

class FraudDetectionModel:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.autoencoder = None
        self.lstm_model = None
        self.hybrid_model = None

    def load_and_preprocess_data(self, file_path):
        print("Loading dataset...")
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            print("Please ensure your file is in CSV format")
            return None, None, None

        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")

        if 'is_fraud' in df.columns:
            print(f"Fraud cases: {df['is_fraud'].sum()} ({df['is_fraud'].mean()*100:.2f}%)")
        else:
            print("Warning: 'is_fraud' column not found!")
            return None, None, None

        print(f"Missing values before cleaning: {df.isnull().sum().sum()}")
        df = df.dropna()
        print(f"Dataset shape after cleaning: {df.shape}")

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

        df = df.sort_values(['sender_id', 'timestamp'])
        df['time_since_last_txn'] = df.groupby('sender_id')['timestamp'].diff().dt.total_seconds()
        df['time_since_last_txn'] = df['time_since_last_txn'].fillna(0)

        df['sender_txn_count'] = df.groupby('sender_id').cumcount() + 1
        df['amount_percentile'] = df.groupby('sender_id')['amount'].rank(pct=True)

        categorical_cols = ['sender_account_type', 'recipient_account_type', 'transaction_type']
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
                print(f"Encoded {col}: {le.classes_}")

        feature_cols = [
            'amount', 'hour', 'day_of_week', 'is_weekend',
            'time_since_last_txn', 'sender_txn_count', 'amount_percentile',
            'sender_balance_before'
        ]

        for col in categorical_cols:
            if col in df.columns:
                feature_cols.append(col + '_encoded')

        if 'merchant_category_code' in df.columns:
            df['merchant_category_code'] = df['merchant_category_code'].fillna(-1)
            feature_cols.append('merchant_category_code')

        available_features = [col for col in feature_cols if col in df.columns]
        print(f"Using features: {available_features}")

        X = df[available_features].values
        y = df['is_fraud'].values

        X_scaled = self.scaler.fit_transform(X)

        return X_scaled, y, df

    def build_hybrid_model(self, input_dim):
        print("Building Hybrid Model...")

        input_layer = Input(shape=(input_dim,))

        auto_encoded = Dense(128, activation='relu')(input_layer)
        auto_encoded = Dropout(0.3)(auto_encoded)
        auto_encoded = Dense(64, activation='relu')(auto_encoded)
        auto_encoded = Dropout(0.3)(auto_encoded)
        auto_encoded = Dense(32, activation='relu')(auto_encoded)
        auto_encoded = Dropout(0.2)(auto_encoded)

        dense_path = Dense(256, activation='relu')(input_layer)
        dense_path = Dropout(0.4)(dense_path)
        dense_path = Dense(128, activation='relu')(dense_path)
        dense_path = Dropout(0.3)(dense_path)
        dense_path = Dense(64, activation='relu')(dense_path)
        dense_path = Dropout(0.2)(dense_path)

        combined = tf.keras.layers.concatenate([auto_encoded, dense_path])
        combined = Dense(64, activation='relu')(combined)
        combined = Dropout(0.3)(combined)
        combined = Dense(32, activation='relu')(combined)
        combined = Dropout(0.2)(combined)
        combined = Dense(16, activation='relu')(combined)
        output = Dense(1, activation='sigmoid')(combined)

        model = Model(input_layer, output)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', Precision(), Recall()]
        )

        return model

    def train_hybrid_model(self, X_train, y_train, X_val, y_val, epochs=100):
        print("Training Hybrid Model...")

        self.hybrid_model = self.build_hybrid_model(X_train.shape[1])

        print("\nModel Architecture:")
        self.hybrid_model.summary()

        fraud_count = np.sum(y_train)
        normal_count = len(y_train) - fraud_count
        weight_for_0 = (1 / normal_count) * (len(y_train) / 2.0)
        weight_for_1 = (1 / fraud_count) * (len(y_train) / 2.0)
        class_weight = {0: weight_for_0, 1: weight_for_1}

        print(f"Class weights: Normal={weight_for_0:.4f}, Fraud={weight_for_1:.4f}")

        early_stopping = EarlyStopping(patience=15, restore_best_weights=True, monitor='val_loss', verbose=1)
        checkpoint = ModelCheckpoint('/content/best_fraud_model.h5', save_best_only=True, monitor='val_loss', verbose=1)

        history = self.hybrid_model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=64,
            validation_data=(X_val, y_val),
            class_weight=class_weight,
            callbacks=[early_stopping, checkpoint],
            verbose=1
        )

        return history

    def evaluate_model(self, X_test, y_test):
        print("Evaluating Model...")

        y_pred_proba = self.hybrid_model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int)

        auc_score = roc_auc_score(y_test, y_pred_proba)

        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        print(f"AUC Score: {auc_score:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Fraud'], yticklabels=['Normal', 'Fraud'])
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')

        plt.subplot(1, 3, 2)
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 3, 3)
        plt.hist(y_pred_proba[y_test == 0], bins=50, alpha=0.7, label='Normal', color='blue')
        plt.hist(y_pred_proba[y_test == 1], bins=50, alpha=0.7, label='Fraud', color='red')
        plt.xlabel('Fraud Probability')
        plt.ylabel('Frequency')
        plt.title('Prediction Distribution')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

        return y_pred_proba

    def save_model(self, save_to_drive=True):
        print("Saving model and preprocessors...")
        self.hybrid_model.save('/content/fraud_detection_model.h5')
        joblib.dump(self.scaler, '/content/scaler.pkl')
        joblib.dump(self.label_encoders, '/content/label_encoders.pkl')

        if save_to_drive:
            try:
                self.hybrid_model.save('/content/drive/MyDrive/fraud_detection_model.h5')
                joblib.dump(self.scaler, '/content/drive/MyDrive/scaler.pkl')
                joblib.dump(self.label_encoders, '/content/drive/MyDrive/label_encoders.pkl')
                print("✅ Model saved to Google Drive!")
            except Exception as e:
                print(f"❌ Could not save to Drive: {e}")

        print("✅ Model saved to /content/ directory!")

    def predict_fraud_score(self, transaction_data):
        scaled_data = self.scaler.transform(transaction_data)
        fraud_score = self.hybrid_model.predict(scaled_data)
        return fraud_score

# ===== MAIN TRAINING FUNCTION =====
def train_fraud_detection_model(csv_file_path, epochs=100):
    print("\U0001F680 Starting FinGuard Pro Model Training...")
    print("="*60)

    fraud_model = FraudDetectionModel()
    X, y, df = fraud_model.load_and_preprocess_data(csv_file_path)

    if X is None:
        print("❌ Failed to load data. Please check your file path.")
        return None

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    print(f"\n📊 Dataset Split:")
    print(f"Training set: {X_train.shape[0]} samples ({np.sum(y_train)} fraud)")
    print(f"Validation set: {X_val.shape[0]} samples ({np.sum(y_val)} fraud)")
    print(f"Test set: {X_test.shape[0]} samples ({np.sum(y_test)} fraud)")

    print(f"\n🔥 Training model for {epochs} epochs...")
    history = fraud_model.train_hybrid_model(X_train, y_train, X_val, y_val, epochs)

    print(f"\n📈 Evaluating model performance...")
    fraud_scores = fraud_model.evaluate_model(X_test, y_test)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3)
    if 'precision' in history.history:
        plt.plot(history.history['precision'], label='Training Precision')
        plt.plot(history.history['val_precision'], label='Validation Precision')
        plt.title('Model Precision')
        plt.xlabel('Epoch')
        plt.ylabel('Precision')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()

    fraud_model.save_model(save_to_drive=True)

    print("\n🎉 Training completed successfully!")
    print("Files saved:")
    print("✅ fraud_detection_model.h5 (main model)")
    print("✅ scaler.pkl (feature scaler)")
    print("✅ label_encoders.pkl (categorical encoders)")

    return fraud_model

# ===== COLAB EXECUTION CELL =====
CSV_FILE_PATH = "/content/drive/MyDrive/synthetic_transaction_data.csv"

if __name__ == "__main__":
    print("📋 BEFORE TRAINING:")
    print("1. Make sure your CSV file is uploaded to Google Drive")
    print("2. Update CSV_FILE_PATH variable above")
    print("3. Run this cell to start training!")
    print("\n" + "="*60)

    if os.path.exists(CSV_FILE_PATH):
        print(f"✅ Found dataset at: {CSV_FILE_PATH}")
        trained_model = train_fraud_detection_model(CSV_FILE_PATH, epochs=50)

        if trained_model:
            print("\n🎯 MODEL TRAINING COMPLETE!")
            print("\n📁 Download your trained model files:")
            print("- fraud_detection_model.h5")
            print("- scaler.pkl")
            print("- label_encoders.pkl")
    else:
        print(f"❌ Dataset not found at: {CSV_FILE_PATH}")
        print("Please check the file path and ensure the file is uploaded to Google Drive.")
