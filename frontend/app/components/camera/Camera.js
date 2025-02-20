import { useEffect, useRef } from 'react';

export default function Camera({ onCapture, onCancel, onError }) {
  const videoRef = useRef(null);

  useEffect(() => {
    startCamera();
    return () => stopCamera();
  }, []);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: {
          width: { ideal: 1280, max: 1920 },
          height: { ideal: 720, max: 1080 },
          facingMode: 'user'
        }
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = async () => {
          try {
            await videoRef.current.play();
          } catch (playError) {
            onError('Failed to start video stream. Please try again.');
            stopCamera();
          }
        };

        videoRef.current.onerror = () => {
          onError('Error with video playback. Please try again.');
        };
      }
    } catch (err) {
      handleCameraError(err);
    }
  };

  const handleCameraError = (err) => {
    if (err.name === 'NotAllowedError') {
      onError('Camera access denied. Please allow camera access in your browser settings.');
    } else if (err.name === 'NotFoundError') {
      onError('No camera found. Please ensure your camera is properly connected.');
    } else if (err.name === 'NotReadableError') {
      onError('Camera is in use by another application. Please close other apps using the camera.');
    } else {
      onError(`Failed to access camera: ${err.message}`);
    }
    stopCamera();
  };

  const stopCamera = () => {
    if (videoRef.current?.srcObject) {
      const tracks = videoRef.current.srcObject.getTracks();
      tracks.forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }
  };

  const captureImage = async () => {
    if (!videoRef.current) return;

    try {
      const canvas = document.createElement('canvas');
      canvas.width = videoRef.current.videoWidth;
      canvas.height = videoRef.current.videoHeight;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(videoRef.current, 0, 0);

      canvas.toBlob((blob) => {
        const file = new File([blob], 'camera-capture.jpg', { type: 'image/jpeg' });
        onCapture(file);
      }, 'image/jpeg', 0.8);
    } catch (err) {
      onError('Failed to capture image. Please try again.');
    }
  };

  return (
    <div className="space-y-4 w-full">
      <div className="w-full h-[480px] bg-black rounded-lg overflow-hidden">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          width="1280"
          height="720"
          style={{
            width: '100%',
            height: '100%',
            objectFit: 'cover',
            transform: 'scaleX(-1)',
            backgroundColor: 'black'
          }}
        />
      </div>
      <div className="flex justify-center gap-4">
        <button
          onClick={captureImage}
          className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition"
        >
          Take Photo
        </button>
        <button
          onClick={onCancel}
          className="px-4 py-2 bg-gray-500 text-white rounded-lg hover:bg-gray-600 transition"
        >
          Cancel
        </button>
      </div>
    </div>
  );
} 
