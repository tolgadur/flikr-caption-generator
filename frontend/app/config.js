'use client';

const getBackendUrl = () => {
  // Check if we have an environment variable
  if (process.env.NEXT_PUBLIC_BACKEND_URL) {
    return process.env.NEXT_PUBLIC_BACKEND_URL;
  }
  
  // Fallback for development
  if (process.env.NODE_ENV === 'development') {
    return 'http://localhost:8000';
  }
  
  // Production fallback (though this should be set by env var)
  return 'http://49.13.206.165:8000';
};

export const config = {
  backendUrl: getBackendUrl(),
}; 
