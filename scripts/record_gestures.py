import os
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

# Save templates directly to backend/templates directory
DATA_DIR = Path(__file__).parent.parent / "backend" / "templates"
DATA_DIR.mkdir(parents=True, exist_ok=True)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def normalize_landmarks(coords: np.ndarray) -> np.ndarray:
    """Normalize 21x3 hand landmarks for scale/translation invariance.
    - Translate so wrist (landmark 0) is origin
    - Scale by max L2 distance to keep values ~[-1,1]
    """
    if coords.shape != (21, 3):
        raise ValueError(f"Expected (21,3) landmarks, got {coords.shape}")
    base = coords[0]
    shifted = coords - base
    scale = np.linalg.norm(shifted, axis=1).max()
    if scale < 1e-6:
        scale = 1.0
    return shifted / scale


def draw_button(frame, text, x, y, w, h, color, text_color=(255, 255, 255)):
    """Draw a clickable button on the frame."""
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
    
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    text_x = x + (w - text_size[0]) // 2
    text_y = y + (h + text_size[1]) // 2
    cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
    return (x, y, w, h)


def is_click_in_button(x, y, button_coords):
    """Check if click coordinates are inside button."""
    bx, by, bw, bh = button_coords
    return bx <= x <= bx + bw and by <= y <= by + bh


def mouse_callback(event, x, y, flags, param):
    """Handle mouse clicks."""
    if event == cv2.EVENT_LBUTTONDOWN:
        param['click'] = (x, y)


def sanitize_filename(name: str) -> str:
    """Encode special characters for safe filenames while preserving meaning."""
    # Replace punctuation with text equivalents
    replacements = {
        '?': '_question',
        '!': '_exclamation',
        '.': '_dot',
        ',': '_comma',
        ':': '_colon',
        ';': '_semicolon',
        "'": '',  # Remove apostrophes
        '"': '',  # Remove quotes
    }
    
    for char, replacement in replacements.items():
        name = name.replace(char, replacement)
    
    # Invalid characters for Windows that can't be encoded
    invalid_chars = '<>/\\|*'
    for char in invalid_chars:
        name = name.replace(char, '')
    
    # Replace spaces with underscores
    name = name.replace(' ', '_')
    # Remove any leading/trailing dots or spaces
    name = name.strip('. ')
    return name


def decode_filename(name: str) -> str:
    """Decode filename back to original form with punctuation."""
    # Reverse the encoding
    replacements = {
        '_question': '?',
        '_exclamation': '!',
        '_dot': '.',
        '_comma': ',',
        '_colon': ':',
        '_semicolon': ';',
    }
    
    for encoded, original in replacements.items():
        name = name.replace(encoded, original)
    
    # Replace underscores with spaces
    name = name.replace('_', ' ')
    return name


def main():
    gesture_raw = input("Enter the gesture name (e.g., hello, how are you): ").strip().lower()
    gesture = sanitize_filename(gesture_raw)
    
    if not gesture:
        print("❌ Error: Invalid gesture name!")
        return
    
    if gesture != gesture_raw:
        print(f"ℹ️  Sanitized gesture name: '{gesture_raw}' → '{gesture}'")
    
    try:
        num_samples = int(input("How many samples to average? (e.g., 5): ").strip())
    except Exception:
        num_samples = 5

    print("\n=== INSTRUCTIONS ===")
    print("1. Click 'START RECORDING' button (green)")
    print("2. Perform your gesture clearly")
    print("3. Click 'SAVE SAMPLE' button when ready")
    print("4. Repeat until all samples are collected\n")

    samples = []
    recording = False
    mouse_data = {'click': None}
    
    # Auto-save timer variables
    auto_save_enabled = True  # Set to False to disable auto-save
    auto_save_duration = 6.0  # Hold gesture steady for 6 seconds to auto-save
    recording_start_time = None
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    cv2.namedWindow("Record Gesture")
    cv2.setMouseCallback("Record Gesture", mouse_callback, mouse_data)

    # Cache the last seen hand pose (even if hands leave frame)
    cached_landmarks = None
    num_hands_cached = 0
    
    with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.6, min_tracking_confidence=0.5) as hands:
        while len(samples) < num_samples:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            # Draw hand landmarks and cache them
            if results.multi_hand_landmarks:
                # Draw all detected hands
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Cache the landmarks for saving later (even if hands leave frame)
                cached_landmarks = results.multi_hand_landmarks
                num_hands_cached = len(results.multi_hand_landmarks)

            # Draw header with gesture info
            cv2.rectangle(frame, (0, 0), (w, 90), (40, 40, 40), -1)
            cv2.putText(frame, f"Gesture: {gesture.upper()}", (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(frame, f"Samples Collected: {len(samples)}/{num_samples}", (15, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

            # Draw recording status - BIG and CLEAR
            hand_detected = results.multi_hand_landmarks is not None
            hands_info = ""
            if hand_detected:
                num_hands = len(results.multi_hand_landmarks)
                hands_info = f" ({num_hands} hand{'s' if num_hands > 1 else ''})"
            
            if recording:
                # Auto-save logic: count down timer when hands are detected
                if auto_save_enabled and (hand_detected or cached_landmarks):
                    if recording_start_time is None:
                        recording_start_time = time.time()
                    
                    elapsed = time.time() - recording_start_time
                    remaining = auto_save_duration - elapsed
                    
                    # Auto-save when timer reaches 0
                    if remaining <= 0:
                        # Trigger auto-save (same as clicking SAVE button)
                        if cached_landmarks:
                            all_coords = []
                            for hand_landmarks in cached_landmarks:
                                coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)
                                coords = normalize_landmarks(coords)
                                all_coords.append(coords)
                            
                            if len(all_coords) == 1:
                                gesture_data = all_coords[0]
                            else:
                                gesture_data = np.vstack(all_coords)
                            
                            samples.append(gesture_data)
                            print(f"✅ AUTO-SAVED Sample {len(samples)}/{num_samples}! ({len(cached_landmarks)} hand{'s' if len(cached_landmarks) > 1 else ''})")
                            recording = False
                            recording_start_time = None
                else:
                    # Reset timer if no hands detected
                    recording_start_time = None
                
                # Blinking RED "RECORDING" indicator
                blink = int(time.time() * 2) % 2  # Blinks every 0.5 seconds
                if blink:
                    cv2.circle(frame, (w - 150, 40), 15, (0, 0, 255), -1)
                    cv2.putText(frame, "RECORDING", (w - 120, 50),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # Hand detection indicator (or show cached hands)
                if hand_detected:
                    hand_status = f"✓ HAND DETECTED{hands_info}"
                    hand_color = (0, 255, 0)
                elif cached_landmarks:
                    hand_status = f"✓ CACHED ({num_hands_cached} hand{'s' if num_hands_cached > 1 else ''})"
                    hand_color = (0, 200, 255)  # Orange - cached
                else:
                    hand_status = "✗ NO HAND - Show your hand!"
                    hand_color = (0, 0, 255)
                
                cv2.putText(frame, hand_status, (15, h - 140),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, hand_color, 2)
                
                # Auto-save countdown timer display
                if auto_save_enabled and recording_start_time is not None and (hand_detected or cached_landmarks):
                    elapsed = time.time() - recording_start_time
                    remaining = max(0, auto_save_duration - elapsed)
                    progress = min(100, int((elapsed / auto_save_duration) * 100))
                    
                    # Big countdown timer
                    timer_text = f"AUTO-SAVE IN: {remaining:.1f}s"
                    cv2.putText(frame, timer_text, (15, h - 100),
                              cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
                    
                    # Progress bar
                    bar_width = 400
                    bar_x = 15
                    bar_y = h - 70
                    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + 30), (60, 60, 60), -1)
                    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_width * progress / 100), bar_y + 30), (0, 255, 255), -1)
                    cv2.putText(frame, f"{progress}%", (bar_x + bar_width + 10, bar_y + 22),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                else:
                    # Instruction
                    if hand_detected or cached_landmarks:
                        cv2.putText(frame, "HOLD STEADY - Auto-save in 3 seconds!", (15, h - 100),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    else:
                        cv2.putText(frame, "Show your hand first!", (15, h - 100),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)
            else:
                recording_start_time = None  # Reset timer when not recording
                cv2.putText(frame, "Click 'START RECORDING' to begin", (15, h - 100),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Draw buttons at bottom (optional - auto-save also works!)
            button_y = h - 40 if recording and recording_start_time is not None else h - 70
            start_btn_color = (0, 200, 0) if not recording else (80, 80, 80)
            # SAVE button is bright green when recording AND we have hand data (current or cached)
            has_hand_data = hand_detected or (cached_landmarks is not None)
            save_btn_color = (0, 255, 0) if (recording and has_hand_data) else (80, 80, 80)
            
            start_btn = draw_button(frame, "START RECORDING", 20, button_y, 280, 55, 
                                   start_btn_color, (255, 255, 255) if not recording else (150, 150, 150))
            # Show manual save button text differently
            manual_save_text = "MANUAL SAVE" if auto_save_enabled else "SAVE SAMPLE"
            save_btn = draw_button(frame, manual_save_text, w - 300, button_y, 280, 55, 
                                  save_btn_color, (255, 255, 255) if (recording and has_hand_data) else (150, 150, 150))
            
            cv2.imshow("Record Gesture", frame)

            # Handle mouse clicks
            if mouse_data['click']:
                click_x, click_y = mouse_data['click']
                
                if is_click_in_button(click_x, click_y, start_btn) and not recording:
                    recording = True
                    recording_start_time = None  # Reset timer
                    print("\n🔴 RECORDING STARTED - Perform your gesture now!")
                    print("   💡 AUTO-SAVE: Hold gesture steady for 3 seconds")
                    print("   💡 MANUAL: Click 'MANUAL SAVE' button anytime")
                    print("   💡 Works with 1 or 2 hands!")
                
                elif is_click_in_button(click_x, click_y, save_btn):
                    if not recording:
                        print("⚠️  Click 'START RECORDING' first!")
                    elif cached_landmarks:
                        # Process all detected hands (1 or 2)
                        all_coords = []
                        for hand_landmarks in cached_landmarks:
                            coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)
                            coords = normalize_landmarks(coords)
                            all_coords.append(coords)
                        
                        # Stack hands: if 1 hand, shape is (21,3); if 2 hands, shape is (42,3)
                        if len(all_coords) == 1:
                            gesture_data = all_coords[0]  # Single hand: (21, 3)
                        else:
                            gesture_data = np.vstack(all_coords)  # Two hands: (42, 3)
                        
                        samples.append(gesture_data)
                        print(f"✅ MANUAL SAVE: Sample {len(samples)}/{num_samples} SAVED! ({len(cached_landmarks)} hand{'s' if len(cached_landmarks) > 1 else ''})")
                        recording = False
                        recording_start_time = None  # Reset timer
                        # Keep cached_landmarks so user can save same gesture multiple times
                    else:
                        print("⚠️  No hand detected or cached! Show your hand clearly in the frame first.")
                        print("   Look for green hand landmarks on screen.")
                
                mouse_data['click'] = None

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n❌ Recording cancelled.")
                break

    cap.release()
    cv2.destroyAllWindows()

    if samples:
        # Filter out samples with inconsistent shapes
        shapes = [arr.shape for arr in samples]
        most_common_shape = max(set(shapes), key=shapes.count)
        
        # Keep only samples with the most common shape
        filtered_samples = [arr for arr in samples if arr.shape == most_common_shape]
        
        if len(filtered_samples) != len(samples):
            print(f"\n⚠️  Warning: Filtered out {len(samples) - len(filtered_samples)} samples with inconsistent hand count")
            print(f"   Kept {len(filtered_samples)} samples with shape {most_common_shape}")
        
        if filtered_samples:
            avg = np.mean(np.stack(filtered_samples, axis=0), axis=0)
            out_path = DATA_DIR / f"{gesture}.npy"
            np.save(out_path, avg)
            num_hands = 2 if avg.shape[0] == 42 else 1
            print(f"\n✅ SUCCESS! Template saved: {out_path}")
            print(f"   Shape: {avg.shape} ({num_hands} hand{'s' if num_hands > 1 else ''})")
            print(f"   You can now record another gesture or run: py scripts\\detect_gesture.py\n")
        else:
            print("❌ No valid samples to save!")
    else:
        print("❌ No samples saved. Nothing written.")


if __name__ == "__main__":
    main()
