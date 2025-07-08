import pickle

SCALER_PATH = "C:/Users/somas/PycharmProjects/FinGuardPro/models/scaler.pkl"

# Load the saved scaler
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

# Check if scaler has feature_names_in_ attribute
if hasattr(scaler, "feature_names_in_"):
    feature_names = scaler.feature_names_in_
    print("Scaler feature names (expected by the model):")
    for i, feat in enumerate(feature_names):
        print(f"{i+1}. {feat}")
else:
    print("Scaler does not store feature names. You may need to check your training data columns.")
