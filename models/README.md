# Models Directory

This directory should contain the trained fraud detection model files:

## Required Files:

1. **fraud_detection_model.h5** - The trained TensorFlow/Keras model
2. **scaler.pkl** - The feature scaler (StandardScaler or similar)
3. **label_encoders.pkl** - Label encoders for categorical features (if used)

## Model Training:

To train the fraud detection model, use the notebook in the `notebooks/` directory or run the training script.

## File Structure:
```
models/
├── fraud_detection_model.h5    # Main model file
├── scaler.pkl                  # Feature scaler
├── label_encoders.pkl          # Label encoders
└── README.md                   # This file
```

## Notes:
- The model expects preprocessed numerical features
- All features should be scaled using the same scaler used during training
- The model outputs fraud probability scores between 0 and 1
- Thresholds: < 0.3 (Low Risk), 0.3-0.7 (Medium Risk), > 0.7 (High Risk)