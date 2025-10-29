"""
Template-based gesture matching using Euclidean distance.
Recognizes words and phrases from pre-recorded templates.
"""

from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np


class TemplateMatcher:
    """Matches hand landmarks against pre-recorded templates using weighted Euclidean distance."""
    
    def __init__(self, templates_dir: str = "templates", threshold: float = 15.0):
        """
        Initialize the template matcher.
        
        Args:
            templates_dir: Directory containing .npy template files
            threshold: Maximum distance for a match (higher = more lenient)
        """
        self.templates_dir = Path(templates_dir)
        self.threshold = threshold
        self.templates: Dict[str, np.ndarray] = {}
        
        # Weight vector: higher weights for fingertips and finger joints
        # Landmark indices: 0=wrist, 4=thumb_tip, 8=index_tip, 12=middle_tip, 16=ring_tip, 20=pinky_tip
        self.weights = np.ones(21)
        self.weights[4] = 3.0   # Thumb tip
        self.weights[8] = 3.0   # Index tip
        self.weights[12] = 3.0  # Middle tip
        self.weights[16] = 3.0  # Ring tip
        self.weights[20] = 3.0  # Pinky tip
        self.weights[2:4] = 2.0   # Thumb joints
        self.weights[5:8] = 2.0   # Index joints
        self.weights[9:12] = 2.0  # Middle joints
        self.weights[13:16] = 2.0 # Ring joints
        self.weights[17:20] = 2.0 # Pinky joints
        
        self.load_templates()
    
    def normalize_landmarks(self, coords: np.ndarray) -> np.ndarray:
        """
        Normalize hand landmarks for scale/translation invariance.
        - Translate so wrist (landmark 0) is origin
        - Scale by max L2 distance
        
        Args:
            coords: Array of shape (21, 3) for single hand or (42, 3) for two hands
            
        Returns:
            Normalized coordinates
        """
        if coords.shape == (21, 3):
            # Single hand
            base = coords[0]
            shifted = coords - base
            scale = np.linalg.norm(shifted, axis=1).max()
            if scale < 1e-6:
                scale = 1.0
            return shifted / scale
        
        elif coords.shape == (42, 3):
            # Two hands - normalize each separately
            hand1 = coords[:21]
            hand2 = coords[21:]
            
            norm_hand1 = self.normalize_landmarks(hand1)
            norm_hand2 = self.normalize_landmarks(hand2)
            
            return np.vstack([norm_hand1, norm_hand2])
        
        else:
            raise ValueError(f"Expected (21,3) or (42,3) landmarks, got {coords.shape}")
    
    def decode_filename(self, name: str) -> str:
        """
        Decode filename back to original form with punctuation.
        
        Args:
            name: Encoded filename (e.g., "what's_up_question")
            
        Returns:
            Decoded string (e.g., "what's up?")
        """
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
    
    def load_templates(self) -> None:
        """Load all .npy template files from the templates directory."""
        self.templates.clear()
        
        if not self.templates_dir.exists():
            print(f"⚠ Templates directory not found: {self.templates_dir}")
            return
        
        loaded_count = 0
        for file in self.templates_dir.glob("*.npy"):
            try:
                gesture = file.stem
                arr = np.load(file).astype(np.float32)
                
                # Accept both single hand (21,3) and two hands (42,3)
                if arr.shape == (21, 3) or arr.shape == (42, 3):
                    self.templates[gesture] = arr
                    loaded_count += 1
                else:
                    print(f"⚠ Invalid template shape {arr.shape} for {gesture}, skipping")
            except Exception as e:
                print(f"⚠ Error loading template {file.name}: {e}")
        
        if loaded_count > 0:
            print(f"[OK] Loaded {loaded_count} templates: {', '.join(sorted(self.templates.keys()))}")
        else:
            print(f"⚠ No templates loaded from {self.templates_dir}")
    
    def match_gesture(self, current: np.ndarray) -> Tuple[Optional[str], float, float]:
        """
        Find the best matching template for current landmarks.
        
        Args:
            current: Normalized landmarks array (21,3) or (42,3)
            
        Returns:
            Tuple of (gesture_name, distance, confidence)
            - gesture_name: Name of matched gesture or None if no match
            - distance: Euclidean distance to best match
            - confidence: Confidence score (0.0 to 1.0)
        """
        if len(self.templates) == 0:
            return None, float("inf"), 0.0
        
        best_name = None
        best_dist = float("inf")
        
        # Prepare weight vector for single or two-hand gestures
        if current.shape[0] == 42:  # Two hands
            weights_full = np.concatenate([self.weights, self.weights])
        else:  # Single hand
            weights_full = self.weights
        
        for name, tmpl in self.templates.items():
            # Only compare if shapes match (both single or both two-hand)
            if current.shape == tmpl.shape:
                # Calculate weighted Euclidean distance
                diff = current - tmpl
                weighted_diff = diff * weights_full[:, np.newaxis]
                dist = np.sqrt(np.sum(weighted_diff ** 2))
                
                if dist < best_dist:
                    best_dist = dist
                    best_name = name
        
        # Calculate confidence based on distance threshold
        if best_name is not None and best_dist < self.threshold:
            confidence = max(0.0, 1.0 - (best_dist / self.threshold))
            return best_name, best_dist, confidence
        
        return None, best_dist, 0.0
    
    def predict(self, landmarks: np.ndarray) -> dict:
        """
        Predict gesture from raw MediaPipe landmarks.
        
        Args:
            landmarks: Flattened array of landmarks [x1,y1,z1, x2,y2,z2, ...]
                      Length 63 for single hand or 126 for two hands
        
        Returns:
            Dictionary with:
            - detected: bool, whether a template matched
            - phrase: str or None, decoded phrase/word
            - distance: float, raw distance to best match
            - confidence: float, confidence score 0-1
            - raw_name: str or None, encoded template name
        """
        # Reshape to (21,3) or (42,3)
        if len(landmarks) == 63:
            coords = landmarks.reshape(21, 3)
        elif len(landmarks) == 126:
            coords = landmarks.reshape(42, 3)
        else:
            return {
                'detected': False,
                'phrase': None,
                'distance': float('inf'),
                'confidence': 0.0,
                'raw_name': None
            }
        
        # Normalize
        try:
            normalized = self.normalize_landmarks(coords)
        except Exception as e:
            print(f"⚠ Normalization error: {e}")
            return {
                'detected': False,
                'phrase': None,
                'distance': float('inf'),
                'confidence': 0.0,
                'raw_name': None
            }
        
        # Match against templates
        name, distance, confidence = self.match_gesture(normalized)
        
        if name is not None:
            decoded_phrase = self.decode_filename(name)
            return {
                'detected': True,
                'phrase': decoded_phrase,
                'distance': float(distance),
                'confidence': float(confidence),
                'raw_name': name
            }
        
        return {
            'detected': False,
            'phrase': None,
            'distance': float(distance),
            'confidence': 0.0,
            'raw_name': None
        }
    
    def get_template_count(self) -> int:
        """Return the number of loaded templates."""
        return len(self.templates)
    
    def get_template_names(self) -> list:
        """Return list of all template names (decoded)."""
        return [self.decode_filename(name) for name in sorted(self.templates.keys())]
