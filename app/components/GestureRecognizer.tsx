'use client';

import { useEffect, useRef, useState, useCallback } from 'react';

interface PredictionResult {
  detected: boolean;
  text: string | null;  // Unified: either letter or phrase
  confidence: number;
  type?: 'letter' | 'phrase' | 'none' | 'unknown';
}

export default function GestureRecognizer() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [prediction, setPrediction] = useState<PredictionResult>({
    detected: false,
    text: null,
    confidence: 0,
    type: 'none',
  });
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [lastSpokenText, setLastSpokenText] = useState<string | null>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  // Feature flag: keep template code present but disabled in the UI/logic when false
  // Set to `true` to re-enable template/phrase recognition in the frontend.
  const USE_TEMPLATE = false;

  // Initialize webcam
  const startWebcam = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 },
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setIsStreaming(true);
      }
    } catch (error) {
      console.error('Error accessing webcam:', error);
      alert('Could not access webcam. Please ensure you have granted camera permissions.');
    }
  };

  // Stop webcam
  const stopWebcam = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach((track) => track.stop());
      videoRef.current.srcObject = null;
      setIsStreaming(false);
    }
  };

  // Capture frame and send to backend
  const captureAndPredict = useCallback(async () => {
    if (!videoRef.current || !canvasRef.current || !isStreaming) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');

    if (!context) return;

    // Draw video frame to canvas
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Convert canvas to base64
    const imageData = canvas.toDataURL('image/jpeg', 0.8);

    try {
      // Send to backend
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image: imageData }),
      });

      if (response.ok) {
        const result: PredictionResult = await response.json();

        // If template/phrase matching is disabled in the frontend, ignore phrase results
        // Keep the template code present in the file, but make it non-functional here.
        if (!USE_TEMPLATE && result.type === 'phrase') {
          // do not surface phrase/template matches to the UI for now
          setPrediction({ detected: false, text: null, confidence: 0, type: 'none' });
        } else {
          setPrediction(result);

          // Speak text if detected with high confidence (works for both letters and phrases)
          if (
            result.detected &&
            result.text &&
            result.confidence > 0.8 &&
            result.text !== lastSpokenText
          ) {
            speakText(result.text);
            setLastSpokenText(result.text);
          }
        }
      }
    } catch (error) {
      console.error('Error predicting gesture:', error);
    }
  }, [isStreaming, lastSpokenText]);

  // Text-to-speech function (works for both letters and phrases)
  const speakText = (text: string) => {
    if ('speechSynthesis' in window && !isSpeaking) {
      setIsSpeaking(true);
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 1.0;
      utterance.pitch = 1.0;
      utterance.volume = 1.0;
      
      utterance.onend = () => {
        setIsSpeaking(false);
      };
      
      window.speechSynthesis.speak(utterance);
    }
  };



  // Start/stop prediction loop
  useEffect(() => {
    if (isStreaming) {
      intervalRef.current = setInterval(captureAndPredict, 200); // 5 FPS
    } else {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [isStreaming, captureAndPredict]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopWebcam();
    };
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-8">
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-5xl font-bold text-gray-900 mb-2">
            ASL Sign Language Translator
          </h1>
          <p className="text-xl text-gray-700">
            Real-time Context-Aware Sign to Text and Speech
          </p>
          <div className="mt-2 text-sm text-gray-600">
            <span className="inline-block bg-green-100 text-green-800 px-3 py-1 rounded-full">
              Model Accuracy: 96.64%
            </span>
            <span className="ml-2 inline-block bg-blue-100 text-blue-800 px-3 py-1 rounded-full">
              Recognizes: A-Z
            </span>

            {/* Template/phrase matching status (kept in file but can be disabled) */}
            {!USE_TEMPLATE ? (
              <span className="ml-2 inline-block bg-red-100 text-red-800 px-3 py-1 rounded-full">
                Template: Disabled
              </span>
            ) : (
              <span className="ml-2 inline-block bg-green-100 text-green-800 px-3 py-1 rounded-full">
                Template: Enabled
              </span>
            )}
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Video Feed */}
          <div className="bg-white rounded-2xl shadow-xl p-6">
            <h2 className="text-2xl font-semibold text-gray-800 mb-4">
              📹 Camera Feed
            </h2>
            <div className="relative bg-black rounded-lg overflow-hidden">
              <video
                ref={videoRef}
                autoPlay
                playsInline
                className="w-full h-auto"
                style={{ maxHeight: '400px' }}
              />
              {!isStreaming && (
                <div className="absolute inset-0 flex items-center justify-center bg-gray-900 bg-opacity-75">
                  <p className="text-white text-lg">Camera Off</p>
                </div>
              )}
            </div>
            <canvas ref={canvasRef} className="hidden" />
            
            <div className="mt-4 flex gap-3">
              {!isStreaming ? (
                <button
                  onClick={startWebcam}
                  className="flex-1 bg-green-600 hover:bg-green-700 text-white font-semibold py-3 px-6 rounded-lg transition-colors duration-200"
                >
                  🎥 Start Camera
                </button>
              ) : (
                <button
                  onClick={stopWebcam}
                  className="flex-1 bg-red-600 hover:bg-red-700 text-white font-semibold py-3 px-6 rounded-lg transition-colors duration-200"
                >
                  Stop Camera
                </button>
              )}
            </div>
          </div>

          {/* Prediction Results */}
          <div className="bg-white rounded-2xl shadow-xl p-6">
            <h2 className="text-2xl font-semibold text-gray-800 mb-4">
              🔍 Recognition Results
            </h2>
            
            <div className="mb-6">
              {prediction.detected ? (
                <div className="text-center">
                  <p className="text-sm text-gray-500 mb-2 uppercase tracking-wide">
                    {prediction.type === 'phrase' ? 'Phrase/Word' : 'Letter'}
                  </p>
                  <div className={`font-bold mb-2 ${
                    prediction.type === 'phrase' 
                      ? 'text-5xl text-green-600' 
                      : 'text-8xl text-indigo-600'
                  }`}>
                    {prediction.text}
                  </div>
                  <div className="flex items-center justify-center gap-2 text-sm text-gray-600">
                    <span className={`inline-block px-3 py-1 rounded-full ${
                      prediction.type === 'phrase'
                        ? 'bg-green-100 text-green-800'
                        : 'bg-indigo-100 text-indigo-800'
                    }`}>
                      {(prediction.confidence * 100).toFixed(0)}% confident
                    </span>
                  </div>
                </div>
              ) : (
                <div className="text-center py-12">
                  <div className="text-6xl mb-4">👋</div>
                  <p className="text-xl text-gray-500">
                    {isStreaming
                      ? 'Show a hand sign...'
                      : 'Start camera to begin'}
                  </p>
                </div>
              )}
            </div>


          </div>
        </div>

        {/* Instructions */}
        <div className="mt-6 bg-white rounded-2xl shadow-xl p-6">
          <h3 className="text-xl font-semibold text-gray-800 mb-3">
            How to Use
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-gray-700">
            <div className="flex items-start gap-3">
              <span className="text-2xl">1️⃣</span>
              <div>
                <p className="font-medium">Start Camera</p>
                <p className="text-sm text-gray-600">
                  Click "Start Camera" to enable webcam
                </p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <span className="text-2xl">2️⃣</span>
              <div>
                <p className="font-medium">Show Hand Sign</p>
                <p className="text-sm text-gray-600">
                  Make ASL alphabet signs (A-Z) in front of the camera
                </p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <span className="text-2xl">3️⃣</span>
              <div>
                <p className="font-medium">View Results</p>
                <p className="text-sm text-gray-600">
                  Letters are recognized and spoken automatically
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
