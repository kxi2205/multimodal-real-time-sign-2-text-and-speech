"""
Flask API for real-time ASL gesture recognition.
Provides endpoints for webcam streaming and gesture prediction.
Supports both alphabet-level (A-Z) and phrase-level (template matching) recognition.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
from pathlib import Path
import cv2
import mediapipe as mp
import base64
from collections import deque
import os
from template_matcher import TemplateMatcher

app = Flask(__name__)
CORS(app)

# Global variables
model = None
scaler = None
labels = None
mp_hands = None
template_matcher = None
prediction_buffer = deque(maxlen=10)

def load_model_and_scaler():
    """Load the trained model and scaler for ASL alphabet recognition."""
    global model, scaler, labels
    
    model_dir = Path(__file__).parent / 'models'
    
    # Load model
    model_path = model_dir / 'model_1.pkl'
    if not model_path.exists():
        raise ValueError(f"Model not found at {model_path}")
    
    print(f"[INFO] Loading model from {model_path}...")
    print("[INFO] This may take a moment on first load...")
    model = joblib.load(model_path)
    print(f"[OK] Loaded model successfully")
    
    # Load scaler
    scaler_path = model_dir / 'scaler_1.pkl'
    if not scaler_path.exists():
        raise ValueError(f"Scaler not found at {scaler_path}")
    
    scaler = joblib.load(scaler_path)
    print(f"[OK] Loaded scaler successfully")
    
    # Create class labels (A-Z)
    labels = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
    print(f"[OK] Labels loaded: {len(labels)} classes (A-Z)")

def initialize_mediapipe():
    """Initialize MediaPipe hands detector."""
    global mp_hands
    
    mp_hands = mp.solutions.hands
    print("[OK] MediaPipe initialized")

def initialize_template_matcher():
    """Initialize template matcher for phrase recognition."""
    global template_matcher
    
    try:
        template_matcher = TemplateMatcher(
            templates_dir=str(Path(__file__).parent / 'templates'),
            threshold=15.0  # Distance threshold for matching
        )
        if template_matcher.get_template_count() > 0:
            print(f"[OK] Template matcher initialized with {template_matcher.get_template_count()} phrases")
        else:
            print("[WARNING] Template matcher initialized but no templates found")
    except Exception as e:
        print(f"[WARNING] Could not initialize template matcher: {e}")
        template_matcher = None

# Initialize on startup
try:
    load_model_and_scaler()
    initialize_mediapipe()
    initialize_template_matcher()
    print("\n[SUCCESS] ASL Recognition Backend Ready!")
    print("[INFO] Alphabet Model Accuracy: 96.64%")
    print("[INFO] Recognizes: A-Z letters + phrases/words")
    if template_matcher:
        print(f"[INFO] Available phrases: {', '.join(template_matcher.get_template_names())}")
    print()
except Exception as e:
    print(f"[ERROR] Error during initialization: {e}")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'alphabet_model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'mediapipe_initialized': mp_hands is not None,
        'template_matcher_loaded': template_matcher is not None,
        'template_count': template_matcher.get_template_count() if template_matcher else 0
    })

@app.route('/predict', methods=['POST'])
def predict_gesture():
    """
    Predict ASL gesture from base64 encoded image.
    Intelligently switches between alphabet and phrase recognition.
    Expected input: {"image": "base64_encoded_image_string"}
    Returns: {"text": "A" or "hello", "confidence": 0.95, "detected": true, "type": "letter" or "phrase"}
    """
    try:
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        # Decode base64 image
        image_data = data['image']
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'error': 'Invalid image data'}), 400
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Use static_image_mode=True for single image processing
        # Set max_num_hands=2 to support two-hand phrases
        with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as hands_detector:
            results = hands_detector.process(rgb_frame)
        
            if results.multi_hand_landmarks:
                # Extract landmarks from all detected hands
                all_landmarks = []
                for hand in results.multi_hand_landmarks:
                    for landmark in hand.landmark:
                        all_landmarks.extend([landmark.x, landmark.y, landmark.z])
                
                landmarks_array = np.array(all_landmarks)
                num_hands = len(results.multi_hand_landmarks)
                
                # === STRATEGY: Prioritize PHRASES (used often), alphabet as fallback (rare) ===
                
                # For SINGLE HAND: Try TEMPLATE/PHRASE FIRST
                if num_hands == 1:
                    # Try template matcher FIRST with lenient threshold
                    if template_matcher:
                        try:
                            phrase_result = template_matcher.predict(landmarks_array)
                            # Use lenient threshold (45%) - phrases are used more often
                            if phrase_result['detected'] and phrase_result['confidence'] >= 0.45:
                                return jsonify({
                                    'detected': True,
                                    'text': phrase_result['phrase'],
                                    'confidence': phrase_result['confidence'],
                                    'type': 'phrase',
                                    'distance': phrase_result['distance']
                                })
                        except Exception as e:
                            print(f"⚠ Template matching error: {e}")
                    
                    # No phrase match? Fall back to alphabet (rare case)
                    # Scale landmarks for alphabet model (expects 63 features)
                    landmarks_scaled = landmarks_array.reshape(1, -1)  # Shape: (1, 63)
                    landmarks_scaled = scaler.transform(landmarks_scaled)
                    
                    # Get prediction
                    prediction = str(model.predict(landmarks_scaled)[0])
                    confidence = float(model.predict_proba(landmarks_scaled).max())
                    
                    # Add to prediction buffer for smoothing
                    prediction_buffer.append((prediction, confidence))
                    
                    # Get smoothed prediction
                    letter = None
                    avg_confidence = 0.0
                    if prediction_buffer:
                        pred_counts = {}
                        conf_sums = {}
                        for pred, conf in prediction_buffer:
                            pred = str(pred)
                            pred_counts[pred] = pred_counts.get(pred, 0) + 1
                            conf_sums[pred] = conf_sums.get(pred, 0) + conf
                        
                        # Get most common prediction
                        most_common = max(pred_counts.items(), key=lambda x: x[1])[0]
                        avg_confidence = conf_sums[most_common] / pred_counts[most_common]
                        letter = most_common
                    
                    # Return alphabet as fallback
                    return jsonify({
                        'detected': True,
                        'text': letter,
                        'confidence': float(avg_confidence),
                        'type': 'letter',
                        'raw_confidence': confidence
                    })
                
                # For TWO HANDS: Try template matcher ONLY (no alphabet)
                elif num_hands == 2:
                    if template_matcher:
                        try:
                            phrase_result = template_matcher.predict(landmarks_array)
                            # Use very lenient threshold (40%) for two-hand phrases
                            if phrase_result['detected'] and phrase_result['confidence'] >= 0.40:
                                return jsonify({
                                    'detected': True,
                                    'text': phrase_result['phrase'],
                                    'confidence': phrase_result['confidence'],
                                    'type': 'phrase',
                                    'distance': phrase_result['distance']
                                })
                        except Exception as e:
                            print(f"⚠ Template matching error: {e}")
                    
                    # No template match for two hands
                    return jsonify({
                        'detected': False,
                        'text': None,
                        'confidence': 0.0,
                        'type': 'unknown',
                        'message': 'Two hands detected but no matching phrase template'
                    })
            
            # No hand detected
            return jsonify({
                'detected': False,
                'text': None,
                'confidence': 0.0,
                'type': 'none'
            })
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/reset', methods=['POST'])
def reset_buffer():
    """Reset the prediction buffer."""
    global prediction_buffer
    prediction_buffer.clear()
    return jsonify({'status': 'buffer reset'})

@app.route('/info', methods=['GET'])
def get_info():
    """Get information about the recognition system."""
    info = {
        'alphabet_model': {
            'name': 'ASL Alphabet Recognition',
            'accuracy': '96.64%',
            'dataset': 'grassknoted/asl-alphabet',
            'classes': len(labels) if labels else 0,
            'labels': labels if labels else [],
            'buffer_size': len(prediction_buffer),
            'max_buffer_size': prediction_buffer.maxlen
        }
    }
    
    if template_matcher:
        info['phrase_recognition'] = {
            'name': 'Template-based Phrase Matching',
            'method': 'Weighted Euclidean Distance',
            'template_count': template_matcher.get_template_count(),
            'phrases': template_matcher.get_template_names(),
            'threshold': template_matcher.threshold
        }
    
    return jsonify(info)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"\n[SERVER] Starting Flask server on http://localhost:{port}")
    app.run(debug=True, host='0.0.0.0', port=port, use_reloader=False)
