import cv2
import numpy as np

class EmotionRecognizer:
    """
    Lightweight emotion recognizer using OpenCV's face detection.
    Returns basic emotions based on facial features without heavy dependencies.
    """
    def __init__(self, backend="opencv"):
        self.backend = backend
        # Initialize OpenCV face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        # Simple emotion mapping based on basic facial features
        # In production, this would use a trained ML model
        self.emotions = ['neutral', 'happy', 'sad', 'surprise', 'angry']

    def predict_emotion(self, frame):
        """
        Detect and predict a basic emotion from the frame.
        Returns None if no face is detected.
        For now, returns 'neutral' as placeholder.
        """
        try:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            if len(faces) == 0:
                return None
            
            # For now, return 'neutral' as placeholder
            # In production, this would analyze facial features
            # to determine actual emotion using a trained model
            return 'neutral'
            
        except Exception as e:
            print(f"Emotion detection error: {e}")
            return None
