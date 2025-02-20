import { useRef, useState } from 'react';

export default function ImageUploadForm({ 
  onSubmit, 
  onFileChange, 
  onUrlChange, 
  onCameraStart,
  file,
  imageUrl,
  loading 
}) {
  const fileInputRef = useRef(null);
  const [feedback, setFeedback] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit();
  };

  const handleFileSelect = (selectedFile) => {
    if (!selectedFile) return;
    onFileChange(selectedFile);
    setFeedback('Image file selected: ' + selectedFile.name);
  };

  const handleUrlInput = (url) => {
    onUrlChange(url);
    if (url) {
      setFeedback('Image URL entered');
      // Reset file input by recreating it
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    } else {
      setFeedback('');
    }
  };

  const handleCameraClick = () => {
    setFeedback('');
    // Reset file input
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
    onUrlChange('');
    onCameraStart();
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Upload Image
        </label>
        <input
          type="file"
          ref={fileInputRef}
          onChange={(e) => handleFileSelect(e.target.files[0])}
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
          onChange={(e) => handleUrlInput(e.target.value)}
          placeholder="https://example.com/image.jpg"
          className="w-full p-2 border border-gray-300 rounded-lg"
        />
      </div>

      {feedback && (
        <div className="text-sm text-green-600 bg-green-50 p-2 rounded-lg">
          {feedback}
        </div>
      )}

      <button
        type="submit"
        disabled={loading || (!file && !imageUrl)}
        className="w-full py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition disabled:opacity-50"
      >
        {loading ? 'Generating Caption...' : 'Generate Caption'}
      </button>

      <div className="text-center">
        <span className="text-gray-500">OR</span>
      </div>

      <button
        type="button"
        onClick={handleCameraClick}
        className="w-full py-2 bg-green-500 text-white rounded-lg hover:bg-green-600 transition"
      >
        Take a Picture
      </button>
    </form>
  );
} 
