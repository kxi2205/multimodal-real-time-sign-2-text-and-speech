"""
Recreate compatible model files for Python 3.12
This script creates new model and scaler files compatible with the current Python version.
"""

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from pathlib import Path

print("[INFO] Creating compatible model files for Python 3.12...")

# Create a simple Random Forest model (placeholder - will learn from actual data when available)
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)

# Create dummy training data (21 landmarks * 3 coordinates = 63 features)
# This is just to make the model compatible - replace with actual training when data is available
np.random.seed(42)
n_samples = 2600  # 100 samples per letter
n_features = 63  # 21 hand landmarks * 3 (x, y, z)
n_classes = 26  # A-Z

X_dummy = np.random.rand(n_samples, n_features)
y_dummy = np.repeat(range(n_classes), n_samples // n_classes)

# Create and fit scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_dummy)

# Train model
print("[INFO] Training model...")
model.fit(X_scaled, y_dummy)

# Save model and scaler
models_dir = Path(__file__).parent / 'models'
models_dir.mkdir(exist_ok=True)

model_path = models_dir / 'model_1.pkl'
scaler_path = models_dir / 'scaler_1.pkl'

joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)

print(f"[OK] Model saved to {model_path}")
print(f"[OK] Scaler saved to {scaler_path}")
print("[WARNING] This is a dummy model for testing. For actual ASL recognition,")
print("[WARNING] you need to train with real ASL hand landmark data.")
print("\n[SUCCESS] Compatible model files created!")
