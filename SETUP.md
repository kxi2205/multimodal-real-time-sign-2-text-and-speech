# 🤟 Multimodal Real-time Context-Aware Sign-to-Text and Speech Translator

A real-time ASL (American Sign Language) alphabet recognition system that uses computer vision and machine learning to translate hand gestures into text and speech.

## 🎯 Features

- **Real-time Gesture Recognition**: Recognizes ASL alphabet signs (A-Z) with 96.64% accuracy
- **Live Webcam Integration**: Captures and processes video feed in real-time
- **Text-to-Speech**: Automatically converts recognized gestures to speech
- **Modern UI**: Beautiful, responsive interface built with Next.js and Tailwind CSS
- **High Performance**: Uses MediaPipe for efficient hand tracking
- **Context-Aware**: Smooths predictions using a buffer system for better accuracy

## 🏗️ Architecture

- **Frontend**: Next.js 15 + React 19 + TypeScript + Tailwind CSS
- **Backend**: Flask API with MediaPipe and scikit-learn
- **ML Model**: Random Forest classifier (96.64% accuracy)
- **Dataset**: grassknoted/asl-alphabet

## 📋 Prerequisites

- Node.js 20+ and npm
- Python 3.8+
- Webcam
- Modern web browser with WebRTC support

## 🚀 Quick Start

### 1. Install Frontend Dependencies

```bash
npm install
```

### 2. Install Backend Dependencies

```bash
npm run install:backend
```

Or manually:

```bash
cd backend
pip install -r requirements.txt
```

### 3. Run the Application

Start both frontend and backend simultaneously:

```bash
npm run dev
```

This will start:
- Frontend: http://localhost:3000
- Backend API: http://localhost:5000

### Alternative: Run Separately

**Frontend:**
```bash
npm run dev:frontend
```

**Backend:**
```bash
npm run dev:backend
```

## 📖 How to Use

1. Open http://localhost:3000 in your browser
2. Click **"Start Camera"** to enable your webcam
3. Show ASL alphabet hand signs (A-Z) in front of the camera
4. Watch as gestures are recognized, converted to text, and spoken aloud!
5. Use **"Speak Text"** to replay recognized text
6. Use **"Clear"** to reset the recognized text

## 🔧 Project Structure

```
.
├── app/                          # Next.js app directory
│   ├── components/
│   │   └── GestureRecognizer.tsx # Main gesture recognition component
│   ├── page.tsx                  # Home page
│   └── layout.tsx                # App layout
├── backend/                      # Flask backend
│   ├── models/                   # ML model files
│   │   ├── model_1.pkl          # Trained Random Forest model
│   │   └── scaler_1.pkl         # Feature scaler
│   ├── app.py                   # Flask API server
│   └── requirements.txt         # Python dependencies
├── package.json                  # Node.js dependencies
└── README.md                     # This file
```

## 🌐 API Endpoints

### Health Check
```
GET /health
```
Returns backend health status

### Predict Gesture
```
POST /predict
Content-Type: application/json

{
  "image": "base64_encoded_image"
}
```
Returns:
```json
{
  "detected": true,
  "letter": "A",
  "confidence": 0.95
}
```

### Reset Buffer
```
POST /reset
```
Clears the prediction smoothing buffer

### Get Info
```
GET /info
```
Returns model information and statistics

## 🧠 Model Information

- **Type**: Random Forest Classifier
- **Accuracy**: 96.64%
- **Classes**: 26 (A-Z)
- **Features**: 63 (21 hand landmarks × 3 coordinates)
- **Framework**: scikit-learn
- **Hand Tracking**: MediaPipe

## 🎨 Tech Stack

### Frontend
- **Framework**: Next.js 15 with Turbopack
- **UI Library**: React 19
- **Language**: TypeScript
- **Styling**: Tailwind CSS 4
- **Speech Synthesis**: Web Speech API

### Backend
- **Framework**: Flask
- **ML Library**: scikit-learn
- **Computer Vision**: MediaPipe + OpenCV
- **Model Persistence**: joblib
- **CORS**: Flask-CORS

## 🔐 Browser Permissions

The application requires:
- **Camera Access**: To capture video for gesture recognition
- **Audio**: For text-to-speech functionality (automatic)

## 🐛 Troubleshooting

### Camera not working
- Ensure you've granted camera permissions in your browser
- Check if another application is using the camera
- Try refreshing the page

### Backend connection errors
- Ensure the Flask backend is running on port 5000
- Check for CORS issues in browser console
- Verify Python dependencies are installed

### Low prediction accuracy
- Ensure good lighting conditions
- Position your hand clearly in the camera frame
- Wait for the model to "warm up" (prediction buffer fills)
- Make clear, distinct ASL alphabet signs

### Python errors
- Ensure you have Python 3.8 or higher
- Install all requirements: `pip install -r backend/requirements.txt`
- Check for conflicting package versions

## 📝 Development Scripts

- `npm run dev` - Run both frontend and backend
- `npm run dev:frontend` - Run only frontend
- `npm run dev:backend` - Run only backend
- `npm run build` - Build for production
- `npm run start` - Start production server
- `npm run lint` - Lint code
- `npm run install:backend` - Install Python dependencies

## 🎓 ASL Alphabet Reference

The system recognizes all 26 letters of the ASL alphabet. For best results:
- Keep your hand within the camera frame
- Use good lighting
- Make clear, distinct signs
- Hold each sign steady for 1-2 seconds

## 🤝 Contributing

This project integrates a machine learning model trained on the grassknoted/asl-alphabet dataset. Future improvements could include:
- Support for ASL words and phrases
- Multi-hand recognition
- Custom gesture training
- Mobile app version
- Offline mode

## 📄 License

This project is open source and available for educational purposes.

## 🙏 Acknowledgments

- **Dataset**: grassknoted/asl-alphabet
- **Hand Tracking**: Google MediaPipe
- **Icons**: Emoji (native)

## 📞 Support

If you encounter any issues or have questions:
1. Check the troubleshooting section
2. Ensure all dependencies are installed
3. Verify your webcam is working
4. Check browser console for errors

---

**Made with ❤️ for accessibility and inclusivity**

🔗 Happy Signing! 🤟
