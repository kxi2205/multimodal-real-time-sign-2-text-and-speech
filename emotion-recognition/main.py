import cv2
from emotion_recognizer import EmotionRecognizer
from video_stream import VideoStream

def main():
    # Choose detector backend: "opencv", "mtcnn", or "retinaface"
    er = EmotionRecognizer(backend="mtcnn")
    vs = VideoStream()

    print("Starting webcam. Press 'q' to quit.")

    while True:
        frame = vs.get_frame()
        if frame is None:
            break

        # Predict emotion
        emotion = er.predict_emotion(frame)

        # Display on frame
        if emotion:
            cv2.putText(frame, f"Emotion: {emotion}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Emotion Recognition", frame)

        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vs.release()

if __name__ == "__main__":
    main()
