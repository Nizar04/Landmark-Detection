import React, { useState } from 'react';
import { Upload, Camera, Loader2 } from 'lucide-react';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';

const LandmarkDetector = () => {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setPreview(URL.createObjectURL(selectedFile));
      setPredictions(null);
      setError(null);
    }
  };

  const handleSubmit = async () => {
    if (!file) return;

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error('Prediction failed');

      const data = await response.json();
      setPredictions(data.predictions);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto p-6">
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold mb-2">Landmark Detection</h1>
        <p className="text-gray-600">Upload an image to identify landmarks</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="space-y-4">
          <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center">
            {preview ? (
              <img 
                src={preview} 
                alt="Preview" 
                className="max-h-64 mx-auto rounded-lg"
              />
            ) : (
              <div className="flex flex-col items-center">
                <Camera className="h-12 w-12 text-gray-400 mb-2" />
                <p className="text-gray-500">Upload an image</p>
              </div>
            )}
            
            <input
              type="file"
              accept="image/*"
              onChange={handleFileChange}
              className="hidden"
              id="file-upload"
            />
            <label
              htmlFor="file-upload"
              className="mt-4 inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 cursor-pointer"
            >
              <Upload className="h-4 w-4 mr-2" />
              Choose File
            </label>
          </div>

          <button
            onClick={handleSubmit}
            disabled={!file || loading}
            className="w-full py-2 px-4 bg-green-600 text-white rounded-md hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? (
              <div className="flex items-center justify-center">
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                Processing...
              </div>
            ) : (
              'Detect Landmarks'
            )}
          </button>
        </div>

        <div className="space-y-4">
          {error && (
            <Alert variant="destructive">
              <AlertTitle>Error</AlertTitle>
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          {predictions && (
            <div className="border rounded-lg p-4">
              <h2 className="text-xl font-semibold mb-4">Results</h2>
              <div className="space-y-2">
                {predictions.map((pred, idx) => (
                  <div 
                    key={idx}
                    className="flex justify-between items-center p-2 bg-gray-50 rounded"
                  >
                    <span className="font-medium">
                      {pred.landmark_id}
                    </span>
                    <span className="text-gray-600">
                      {(pred.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default LandmarkDetector;
