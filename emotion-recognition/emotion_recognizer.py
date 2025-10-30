from deepface import DeepFace
import cv2
class EmotionRecognizer:
    def __init__(self, backend="opencv"):
        self.backend = backend

    def predict_emotion(self, frame):
        """
        Detect and predict the dominant emotion in the given frame.
        Returns None if no face is detected.
        """
        try:
            result = DeepFace.analyze(
                frame,
                actions=['emotion'],
                detector_backend=self.backend,
                enforce_detection=False
            )
            return result[0]['dominant_emotion']
        except:
            return None
