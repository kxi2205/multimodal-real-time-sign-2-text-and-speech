# ML Models Directory

## Required Model Files

Due to GitHub's file size limitations (100 MB max), the trained model files are not included in this repository. You need to download or train them separately.

### Required Files:
- `model_1.pkl` - Random Forest classifier (96.64% accuracy, 26 classes A-Z)
- `scaler_1.pkl` - StandardScaler for feature normalization

### File Size:
- Each file is approximately 213 MB

## Option 1: Download Pre-trained Models

If you have access to the original model files, place them in this directory:
```
backend/models/
├── model_1.pkl
├── scaler_1.pkl
└── README.md (this file)
```

## Option 2: Train Your Own Model

Use the training script from the original minor project or train a new model:

1. Collect ASL hand gesture dataset (e.g., from Kaggle: grassknoted/asl-alphabet)
2. Extract MediaPipe hand landmarks (21 landmarks × 3 coordinates = 63 features)
3. Train a Random Forest classifier
4. Use StandardScaler for feature normalization
5. Save both model and scaler using `joblib`

Example training code:
```python
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# After training...
joblib.dump(model, 'model_1.pkl')
joblib.dump(scaler, 'scaler_1.pkl')
```

## Model Specifications

- **Algorithm**: Random Forest Classifier
- **Features**: 63 (21 MediaPipe hand landmarks × 3 coordinates each)
- **Classes**: 26 (A-Z letters)
- **Accuracy**: 96.64%
- **Dataset**: grassknoted/asl-alphabet
- **Framework**: scikit-learn 1.3.2
- **Python Version**: 3.12

## Verification

After placing the model files, verify they work:
```bash
cd backend
python -c "import joblib; model = joblib.load('models/model_1.pkl'); print('Model loaded successfully')"
```

## Note

The `backup/` subdirectory may contain backup versions of the models from conversion processes. These are also excluded from git.
