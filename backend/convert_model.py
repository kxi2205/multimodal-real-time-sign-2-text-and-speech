"""
Convert old model files to Python 3.12 compatible format
This script loads the model with compatibility mode and resaves it
"""

import sys
import warnings
warnings.filterwarnings('ignore')

print("[INFO] Attempting to load and convert model files...")

# Try to load with older Python compatibility
try:
    import joblib
    import numpy as np
    from pathlib import Path
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    
    models_dir = Path(__file__).parent / 'models'
    
    # Backup original files
    import shutil
    backup_dir = models_dir / 'backup'
    backup_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / 'model_1.pkl'
    scaler_path = models_dir / 'scaler_1.pkl'
    
    print(f"[INFO] Backing up original files...")
    shutil.copy(model_path, backup_dir / 'model_1.pkl.bak')
    shutil.copy(scaler_path, backup_dir / 'scaler_1.pkl.bak')
    
    print(f"[INFO] Loading model (this may take a while)...")
    # Load with fix_imports for compatibility
    model = joblib.load(model_path)
    print(f"[OK] Model loaded successfully")
    
    print(f"[INFO] Loading scaler...")
    scaler = joblib.load(scaler_path)
    print(f"[OK] Scaler loaded successfully")
    
    print(f"[INFO] Resaving with current Python version...")
    joblib.dump(model, model_path, protocol=4)
    joblib.dump(scaler, scaler_path, protocol=4)
    
    print(f"[SUCCESS] Model files converted successfully!")
    print(f"[INFO] Original files backed up to: {backup_dir}")
    print(f"[INFO] You can now run the backend: py app.py")
    
except Exception as e:
    print(f"[ERROR] Failed to convert model: {e}")
    print(f"\n[SOLUTION] The model was trained with an older Python version.")
    print(f"[SOLUTION] Options:")
    print(f"  1. Install Python 3.10 or 3.11 and use that for the backend")
    print(f"  2. Retrain the model with current Python 3.12")
    print(f"  3. Use a virtual environment with compatible versions")
    sys.exit(1)
