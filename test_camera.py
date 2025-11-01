"""
Real-Time Sign Language Recognition Camera Test
Run this script to test your trained model with your webcam
"""

import torch
import torchvision.transforms as transforms
import timm
import cv2
import json
import numpy as np
from PIL import Image
import time
from pathlib import Path
import argparse
import mediapipe as mp


def load_model(model_path, class_mapping_path, device):
    """Load trained model and class mappings"""
    print("Loading model...")
    
    # Load class mappings
    with open(class_mapping_path, 'r') as f:
        class_mapping = json.load(f)
    
    label_names = class_mapping['label_names']
    num_classes = class_mapping['num_classes']
    
    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model = timm.create_model(checkpoint['model_name'], pretrained=False, num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded successfully!")
    print(f"   - Architecture: {checkpoint['model_name']}")
    print(f"   - Classes: {num_classes}")
    print(f"   - Validation Accuracy: {checkpoint['best_val_acc']:.2%}")
    
    return model, label_names


def get_transform(img_size=224):
    """Get image preprocessing transform"""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def preprocess_frame(frame, transform):
    """Preprocess camera frame for model input"""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    tensor = transform(pil_image).unsqueeze(0)
    return tensor


def detect_hand(frame, hands_detector):
    """
    Detect if a hand is present in the frame using MediaPipe
    Returns: (hand_detected: bool, frame_with_landmarks: frame)
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(frame_rgb)
    
    hand_detected = results.multi_hand_landmarks is not None
    
    # Draw hand landmarks on frame
    if hand_detected:
        mp_drawing = mp.solutions.drawing_utils
        mp_hands = mp.solutions.hands
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
            )
    
    return hand_detected, frame


def draw_predictions(frame, prediction, hand_detected, fps=0):
    """Draw prediction results on frame (single prediction only)"""
    overlay = frame.copy()
    
    # Draw FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Draw prediction box
    box_height = 120
    cv2.rectangle(overlay, (10, 50), (450, 50 + box_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Draw title
    cv2.putText(frame, "Sign Language Detection", (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Draw hand detection status
    hand_status = "Hand Detected ✓" if hand_detected else "No Hand Detected"
    hand_color = (0, 255, 0) if hand_detected else (0, 0, 255)
    cv2.putText(frame, hand_status, (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, hand_color, 2)
    
    # Draw prediction only if hand is detected
    if hand_detected and prediction:
        label = prediction['label']
        confidence = prediction['confidence']
        
        # Color based on confidence
        color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255)
        
        text = f"Sign: {label} ({confidence:.1%})"
        cv2.putText(frame, text, (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    elif hand_detected:
        cv2.putText(frame, "Sign: Processing...", (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    # Draw instructions
    cv2.putText(frame, "Press 'q' to quit | 's' for screenshot", (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return frame


def run_camera(model, label_names, transform, device, camera_index=0, inference_interval=0.1, confidence_threshold=0.5):
    """Run real-time sign language detection with MediaPipe hand detection"""
    print("\n" + "=" * 70)
    print("Starting camera... Press 'q' to quit, 's' for screenshot")
    print("=" * 70)
    
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print("❌ Error: Could not open camera!")
        print("Try changing camera_index (use --camera 1, --camera 2, etc.)")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands_detector = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,  # Detect only one hand at a time
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    print(f"✅ Camera opened successfully!")
    print(f"   Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print(f"✅ MediaPipe hand detection initialized")
    print("\n🤟 Make sign language gestures in front of the camera!")
    print("=" * 70)
    
    last_inference_time = 0
    prediction = None
    frame_count = 0
    start_time = time.time()
    screenshot_count = 0
    hand_detected = False
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to grab frame")
                break
            
            current_time = time.time()
            frame_count += 1
            
            # Detect hand in frame
            hand_detected, frame = detect_hand(frame, hands_detector)
            
            # Run inference only if hand is detected and at specified interval
            if hand_detected and (current_time - last_inference_time >= inference_interval):
                input_tensor = preprocess_frame(frame, transform).to(device)
                
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probs = torch.softmax(outputs, dim=1)[0]
                    top_prob, top_idx = torch.max(probs, dim=0)
                
                # Always show prediction when hand is detected (no threshold)
                prediction = {
                    'label': label_names[top_idx.item()],
                    'confidence': float(top_prob.item())
                }
                
                last_inference_time = current_time
            elif not hand_detected:
                # Clear prediction when no hand is detected
                prediction = None
            
            # Calculate FPS
            fps = frame_count / (current_time - start_time)
            
            # Draw predictions
            annotated_frame = draw_predictions(frame, prediction, hand_detected, fps)
            
            # Display
            cv2.imshow('Sign Language Recognition - Press Q to quit', annotated_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('s'):
                screenshot_count += 1
                filename = f'screenshot_{screenshot_count}.png'
                cv2.imwrite(filename, annotated_frame)
                print(f"📸 Screenshot saved: {filename}")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands_detector.close()
        
        # Print statistics
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        print("\n" + "=" * 70)
        print("Camera Session Summary:")
        print(f"  • Total frames: {frame_count}")
        print(f"  • Duration: {total_time:.1f}s")
        print(f"  • Average FPS: {avg_fps:.1f}")
        print(f"  • Screenshots taken: {screenshot_count}")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Real-time Sign Language Recognition with MediaPipe Hand Detection')
    parser.add_argument('--model', type=str, default='backend/models/sign_language_model.pt',
                        help='Path to trained model file')
    parser.add_argument('--mapping', type=str, default='backend/models/class_mapping.json',
                        help='Path to class mapping JSON file')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera device index (0 for default)')
    parser.add_argument('--interval', type=float, default=0.05,
                        help='Inference interval in seconds (lower = faster but more CPU)')
    parser.add_argument('--threshold', type=float, default=0.0,
                        help='Confidence threshold for predictions (0.0-1.0, 0=disabled)')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU inference (default: use GPU if available)')
    
    args = parser.parse_args()
    
    # Setup device
    if args.cpu:
        device = torch.device('cpu')
        print("Using CPU for inference")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
    
    # Check if files exist
    model_path = Path(args.model)
    mapping_path = Path(args.mapping)
    
    if not model_path.exists():
        print(f"❌ Error: Model file not found: {model_path}")
        print("Please train the model first using the Jupyter notebook.")
        return
    
    if not mapping_path.exists():
        print(f"❌ Error: Class mapping file not found: {mapping_path}")
        print("Please train the model first using the Jupyter notebook.")
        return
    
    # Load model
    model, label_names = load_model(str(model_path), str(mapping_path), device)
    
    # Get transform
    transform = get_transform()
    
    # Run camera with MediaPipe hand detection
    run_camera(model, label_names, transform, device, args.camera, args.interval, args.threshold)


if __name__ == '__main__':
    main()
