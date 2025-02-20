import { useState } from 'react';

export default function ResultModal({ isOpen, onClose, imageUrl, caption, allCaptions, imageFile }) {
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);

  if (!isOpen) return null;

  // Create object URL for file if it exists
  const displayUrl = imageFile ? URL.createObjectURL(imageFile) : imageUrl;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
      <div className="bg-white rounded-xl max-w-3xl w-full max-h-[90vh] overflow-y-auto p-6 relative">
        {/* Close button */}
        <button
          onClick={onClose}
          className="absolute top-4 right-4 text-gray-500 hover:text-gray-700"
        >
          <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>

        <div className="space-y-6">
          {/* Image */}
          <div className="w-full aspect-video bg-gray-100 rounded-lg overflow-hidden">
            <img
              src={displayUrl}
              alt="Uploaded image"
              className="w-full h-full object-contain"
            />
          </div>

          {/* Caption */}
          <div className="space-y-2">
            <h3 className="text-xl font-semibold text-gray-900">Generated Caption</h3>
            <p className="text-lg text-gray-700">{caption}</p>
          </div>

          {/* All Captions Dropdown */}
          <div className="space-y-2">
            <button
              onClick={() => setIsDropdownOpen(!isDropdownOpen)}
              className="flex items-center space-x-2 text-sm text-gray-600 hover:text-gray-800"
            >
              <span>View all generated captions</span>
              <svg
                className={`w-4 h-4 transform transition-transform ${isDropdownOpen ? 'rotate-180' : ''}`}
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </button>
            
            {isDropdownOpen && (
              <div className="bg-gray-50 rounded-lg p-4 space-y-2">
                {allCaptions?.map((cap, index) => (
                  <p key={index} className="text-sm text-gray-600">
                    {cap}
                  </p>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
} 
