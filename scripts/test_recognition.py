"""
Test gesture recognition using the integrated backend system.
This script runs independently to test both alphabet and phrase recognition.
"""

import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

# Add backend to path to import modules
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path.resolve()))

try:
    from template_matcher import TemplateMatcher # type: ignore
    import joblib
    BACKEND_AVAILABLE = True
except ImportError as e:
    print(f"⚠ Could not import backend modules: {e}")
    print(f"Backend path: {backend_path.resolve()}")
    BACKEND_AVAILABLE = False
    sys.exit(1)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Load alphabet model and template matcher
model = None
scaler = None
labels = None
template_matcher = None


def load_backend_models():
    """Load alphabet model and template matcher."""
    global model, scaler, labels, template_matcher
    
    # Load alphabet model
    model_dir = backend_path / 'models'
    model_path = model_dir / 'model_1.pkl'
    scaler_path = model_dir / 'scaler_1.pkl'
    
    if model_path.exists() and scaler_path.exists():
        print("[INFO] Loading alphabet model...")
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        labels = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
        print(f"[OK] Alphabet model loaded (A-Z)")
    else:
        print("⚠ Alphabet model files not found")
    
    # Load template matcher
    templates_dir = backend_path / 'templates'
    if templates_dir.exists():
        print("[INFO] Loading template matcher...")
        template_matcher = TemplateMatcher(str(templates_dir), threshold=15.0)
        print(f"[OK] Template matcher loaded ({template_matcher.get_template_count()} phrases)")
        if template_matcher.get_template_count() > 0:
            print(f"[INFO] Available phrases: {', '.join(template_matcher.get_template_names())}")
    else:
        print("⚠ Templates directory not found")


def predict_gesture(landmarks_array, num_hands):
    """
    Predict gesture using the same logic as backend.
    Prioritizes PHRASES (used often), alphabet as fallback (rare).
    """
    result = {
        'detected': False,
        'text': None,
        'confidence': 0.0,
        'type': 'none'
    }
    
    # For SINGLE HAND: Try TEMPLATE/PHRASE FIRST
    if num_hands == 1:
        # Try template matcher FIRST with lenient threshold
        if template_matcher:
            try:
                phrase_result = template_matcher.predict(landmarks_array)
                # Use lenient threshold (45%) - phrases are used more often
                if phrase_result['detected'] and phrase_result['confidence'] >= 0.45:
                    return {
                        'detected': True,
                        'text': phrase_result['phrase'],
                        'confidence': phrase_result['confidence'],
                        'type': 'phrase',
                        'distance': phrase_result['distance']
                    }
            except Exception as e:
                print(f"Template matching error: {e}")
        
        # No phrase match? Fall back to alphabet (rare case)
        if model:
            try:
                landmarks_scaled = landmarks_array.reshape(1, -1)
                landmarks_scaled = scaler.transform(landmarks_scaled)
                
                prediction = str(model.predict(landmarks_scaled)[0])
                confidence = float(model.predict_proba(landmarks_scaled).max())
                
                return {
                    'detected': True,
                    'text': prediction,
                    'confidence': confidence,
                    'type': 'letter'
                }
            except Exception as e:
                print(f"⚠ Alphabet prediction error: {e}")
    
    # For TWO HANDS: Try template matcher ONLY
    elif template_matcher and num_hands == 2:
        try:
            phrase_result = template_matcher.predict(landmarks_array)
            # Use very lenient threshold (40%) for two-hand phrases
            if phrase_result['detected'] and phrase_result['confidence'] >= 0.40:
                return {
                    'detected': True,
                    'text': phrase_result['phrase'],
                    'confidence': phrase_result['confidence'],
                    'type': 'phrase',
                    'distance': phrase_result['distance']
                }
        except Exception as e:
            print(f"Template matching error: {e}")
    
    return result


def main():
    """Run the test application."""
    if not BACKEND_AVAILABLE:
        print("Backend modules not available. Exiting.")
        return
    
    load_backend_models()
    
    if not model and not template_matcher:
        print("\n⚠ No recognition models loaded!")
        print("Please ensure:")
        print("  1. Alphabet model files exist in backend/models/")
        print("  2. Template files exist in backend/templates/")
        return
    
    print("\n" + "="*60)
    print("  GESTURE RECOGNITION TEST")
    print("="*60)
    print("Press 'Q' to quit\n")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5
    ) as hands:
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            
            # Default values
            text = "None"
            confidence = 0.0
            rec_type = "none"
            num_hands_detected = 0
            
            if results.multi_hand_landmarks:
                # Draw all detected hands
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Extract landmarks
                all_landmarks = []
                for hand in results.multi_hand_landmarks:
                    for landmark in hand.landmark:
                        all_landmarks.extend([landmark.x, landmark.y, landmark.z])
                
                landmarks_array = np.array(all_landmarks)
                num_hands_detected = len(results.multi_hand_landmarks)
                
                # Predict gesture
                result = predict_gesture(landmarks_array, num_hands_detected)
                
                if result['detected']:
                    text = result['text']
                    confidence = result['confidence']
                    rec_type = result['type']
            
            # Render UI
            # Background panel
            cv2.rectangle(frame, (10, 10), (630, 150), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, 10), (630, 150), (255, 255, 255), 2)
            
            # Status
            status_color = (0, 255, 0) if num_hands_detected > 0 else (0, 0, 255)
            cv2.putText(frame, f"Hands: {num_hands_detected}", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            
            # Recognition type
            type_text = rec_type.upper() if rec_type != 'none' else 'WAITING'
            type_color = (0, 255, 0) if rec_type == 'phrase' else (255, 100, 0) if rec_type == 'letter' else (150, 150, 150)
            cv2.putText(frame, f"Type: {type_text}", (20, 75), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, type_color, 2)
            
            # Detected text
            text_color = (0, 255, 0) if rec_type == 'phrase' else (255, 100, 0) if rec_type == 'letter' else (255, 255, 255)
            cv2.putText(frame, f"Detected: {text}", (20, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)
            
            # Confidence
            if confidence > 0:
                conf_text = f"Confidence: {confidence:.0%}"
                cv2.putText(frame, conf_text, (20, 140), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            
            # Instructions
            cv2.putText(frame, "Press [Q] to Quit", (10, 470), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
            
            cv2.imshow("Gesture Recognition Test", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\nTest complete!")


if __name__ == "__main__":
    main()
