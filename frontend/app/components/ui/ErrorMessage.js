export default function ErrorMessage({ message }) {
  if (!message) return null;
  
  return (
    <div className="bg-red-50 text-red-700 p-4 rounded-lg">
      {message}
    </div>
  );
} 
