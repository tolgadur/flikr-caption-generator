'use client';

import { useState } from 'react';
import Camera from './components/camera/Camera';
import ImageUploadForm from './components/form/ImageUploadForm';
import ResultModal from './components/ResultModal';
import ErrorMessage from './components/ui/ErrorMessage';

export default function Home() {
  const [imageUrl, setImageUrl] = useState('');
  const [file, setFile] = useState(null);
  const [caption, setCaption] = useState('');
  const [allCaptions, setAllCaptions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [showModal, setShowModal] = useState(false);
  const [showCamera, setShowCamera] = useState(false);

  const handleSubmit = async (capturedFile = null) => {
    setLoading(true);
    setError('');
    
    try {
      let response;
      const fileToUse = capturedFile || file;
      
      if (fileToUse) {
        const formData = new FormData();
        formData.append('image', fileToUse);
        
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
      setAllCaptions(data.all_captions);
      setShowModal(true);
    } catch (err) {
      setError(err.message || 'Failed to generate caption. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleFileChange = (selectedFile) => {
    if (!selectedFile) return;
    setFile(selectedFile);
    setImageUrl('');
  };

  const handleUrlChange = (url) => {
    setImageUrl(url);
    setFile(null);
  };

  const handleCameraCapture = async (capturedFile) => {
    setShowCamera(false);
    setImageUrl('');
    setFile(capturedFile);
    setError('');
    await handleSubmit(capturedFile);
  };

  return (
    <main className="min-h-screen p-8 flex flex-col items-center justify-center bg-gray-50">
      <div className="w-full max-w-2xl space-y-8">
        <h1 className="text-4xl font-bold text-center text-gray-800 mb-8">
          Image Caption Generator
        </h1>

        {showCamera ? (
          <Camera
            onCapture={handleCameraCapture}
            onCancel={() => setShowCamera(false)}
            onError={setError}
          />
        ) : (
          <div className="bg-white p-8 rounded-xl shadow-lg space-y-6">
            <ImageUploadForm
              onSubmit={handleSubmit}
              onFileChange={handleFileChange}
              onUrlChange={handleUrlChange}
              onCameraStart={() => setShowCamera(true)}
              file={file}
              imageUrl={imageUrl}
              loading={loading}
            />

            <ErrorMessage message={error} />

            <ResultModal
              isOpen={showModal}
              onClose={() => setShowModal(false)}
              imageUrl={imageUrl}
              imageFile={file}
              caption={caption}
              allCaptions={allCaptions}
            />
          </div>
        )}
      </div>
    </main>
  );
}
