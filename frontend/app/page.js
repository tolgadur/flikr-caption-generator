'use client';

import { useState, useRef, useEffect } from 'react';

export default function Home() {
  const [imageUrl, setImageUrl] = useState('');
  const [file, setFile] = useState(null);
  const [caption, setCaption] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const fileInputRef = useRef(null);
  const videoRef = useRef(null);
  const [showCamera, setShowCamera] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    
    try {
      let response;
      
      if (file) {
        const formData = new FormData();
        formData.append('image', file);
        
        response = await fetch('http://localhost:8000/generate-caption', {
          method: 'POST',
          body: formData,
        });
      } else if (imageUrl) {
        response = await fetch('http://localhost:8000/generate-caption-from-url', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ url: imageUrl }),
        });
      } else {
        throw new Error('Please provide an image or URL');
      }
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || 'Failed to generate caption');
      }
      
      const data = await response.json();
      setCaption(data.caption);
    } catch (err) {
      setError(err.message || 'Failed to generate caption. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = (e) => {
    const selectedFile = e.target.files[0];
    if (!selectedFile) return;
    
    setFile(selectedFile);
    setImageUrl(''); // Clear URL when file is selected
  };

  const handleUrlChange = (e) => {
    setImageUrl(e.target.value);
    setFile(null); // Clear file when URL is entered
  };

  const startCamera = async () => {
    try {
      console.log('Requesting camera access...');
      setShowCamera(true);  // Set this first so the video element is mounted
      
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: {
          width: { ideal: 1280, max: 1920 },
          height: { ideal: 720, max: 1080 },
          facingMode: 'user'
        }
      });
      
      console.log('Camera access granted, setting up video element...');
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        
        // Debug video element state
        console.log('Video element ready state:', videoRef.current.readyState);
        console.log('Video element dimensions:', {
          width: videoRef.current.videoWidth,
          height: videoRef.current.videoHeight
        });

        videoRef.current.onloadedmetadata = async () => {
          console.log('Video metadata loaded');
          try {
            await videoRef.current.play();
            console.log('Video playback started');
          } catch (playError) {
            console.error('Error playing video:', playError);
            setError('Failed to start video stream. Please try again.');
            stopCamera();
          }
        };

        // Add error event listener
        videoRef.current.onerror = (error) => {
          console.error('Video element error:', error);
          setError('Error with video playback. Please try again.');
        };
      }
    } catch (err) {
      console.error('Camera access error:', err);
      if (err.name === 'NotAllowedError') {
        setError('Camera access denied. Please allow camera access in your browser settings.');
      } else if (err.name === 'NotFoundError') {
        setError('No camera found. Please ensure your camera is properly connected.');
      } else if (err.name === 'NotReadableError') {
        setError('Camera is in use by another application. Please close other apps using the camera.');
      } else {
        setError(`Failed to access camera: ${err.message}`);
      }
      stopCamera();
    }
  };

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const tracks = videoRef.current.srcObject.getTracks();
      tracks.forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }
    setShowCamera(false);
  };

  const captureImage = async () => {
    if (!videoRef.current) return;

    try {
      const canvas = document.createElement('canvas');
      canvas.width = videoRef.current.videoWidth;
      canvas.height = videoRef.current.videoHeight;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(videoRef.current, 0, 0);

      canvas.toBlob(async (blob) => {
        const file = new File([blob], 'camera-capture.jpg', { type: 'image/jpeg' });
        setFile(file);
        setImageUrl('');
        stopCamera();
        
        // Automatically submit the captured image
        const formData = new FormData();
        formData.append('image', file);
        
        setLoading(true);
        try {
          const response = await fetch('http://localhost:8000/generate-caption', {
            method: 'POST',
            body: formData,
          });
          
          if (!response.ok) throw new Error('Failed to generate caption');
          
          const data = await response.json();
          setCaption(data.caption);
        } catch (err) {
          setError('Failed to generate caption. Please try again.');
        } finally {
          setLoading(false);
        }
      }, 'image/jpeg', 0.8);
    } catch (err) {
      setError('Failed to capture image. Please try again.');
      console.error('Capture error:', err);
    }
  };

  // Cleanup camera on unmount
  useEffect(() => {
    return () => {
      if (showCamera) {
        stopCamera();
      }
    };
  }, [showCamera]);

  return (
    <main className="min-h-screen p-8 flex flex-col items-center justify-center bg-gray-50">
      <div className="w-full max-w-2xl space-y-8">
        <h1 className="text-4xl font-bold text-center text-gray-800 mb-8">
          Image Caption Generator
        </h1>

        {showCamera ? (
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
                onClick={stopCamera}
                className="px-4 py-2 bg-gray-500 text-white rounded-lg hover:bg-gray-600 transition"
              >
                Cancel
              </button>
            </div>
          </div>
        ) : (
          <div className="bg-white p-8 rounded-xl shadow-lg space-y-6">
            <form onSubmit={handleSubmit} className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Upload Image
                </label>
                <input
                  type="file"
                  ref={fileInputRef}
                  onChange={handleFileUpload}
                  accept="image/*"
                  className="w-full p-2 border border-gray-300 rounded-lg"
                />
                {file && (
                  <p className="mt-2 text-sm text-gray-600">
                    Selected file: {file.name}
                  </p>
                )}
              </div>

              <div className="text-center">
                <span className="text-gray-500">OR</span>
              </div>

              <div className="space-y-2">
                <label className="block text-sm font-medium text-gray-700">
                  Image URL
                </label>
                <input
                  type="url"
                  value={imageUrl}
                  onChange={handleUrlChange}
                  placeholder="https://example.com/image.jpg"
                  className="w-full p-2 border border-gray-300 rounded-lg"
                />
              </div>

              <button
                type="submit"
                disabled={loading || (!file && !imageUrl)}
                className="w-full py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition disabled:opacity-50"
              >
                Generate Caption
              </button>

              <div className="text-center">
                <span className="text-gray-500">OR</span>
              </div>

              <button
                type="button"
                onClick={startCamera}
                className="w-full py-2 bg-green-500 text-white rounded-lg hover:bg-green-600 transition"
              >
                Take a Picture
              </button>
            </form>

            {loading && (
              <div className="text-center text-gray-600">
                Generating caption...
              </div>
            )}

            {error && (
              <div className="text-center text-red-500">
                {error}
              </div>
            )}

            {caption && (
              <div className="mt-8 p-4 bg-gray-50 rounded-lg">
                <h2 className="text-xl font-semibold text-center text-gray-800 mb-2">
                  Generated Caption
                </h2>
                <p className="text-lg text-center text-gray-700">
                  {caption}
                </p>
              </div>
            )}
          </div>
        )}
      </div>
    </main>
  );
}
