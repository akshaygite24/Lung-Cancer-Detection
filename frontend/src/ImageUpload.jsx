import React, { useState } from "react";
import axios from "axios";
import "./ImageUpload.css";

const ImageUpload = () => {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [confidence, setConfidence] = useState(null);
  const [feedback, setFeedback] = useState(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      console.log(selectedFile);
      setPreview(URL.createObjectURL(selectedFile));
      console.log(preview);
    }
  };

  const handleSubmit = async () => {
    if (!file) return;
    
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await axios.post("http://127.0.0.1:8000/predict/", formData);
      setPrediction(response.data.prediction);
      setConfidence(response.data.confidence);
      setFeedback(response.data.feedback);
    } catch (error) {
      console.error("Error uploading file", error);
    }
  };

  return (
    <div className="image-upload-container">
      <h2>Lung Cancer Detection</h2>
      <div className="upload-controls">
        <input type="file" onChange={handleFileChange} accept="image/*" />
        <button onClick={handleSubmit} disabled={!file}>Upload & Predict</button>
      </div>
      
      {preview && (
        <div className="image-preview-wrapper">
          <img src={preview} alt="Preview" className="image-preview" />
        </div>
      )}
      
      {prediction && (
        <div className="prediction-results">
          <h3>Prediction: {prediction}</h3>
          <h4>Confidence: {(confidence * 100).toFixed(2)}%</h4>
          <p><strong>Feedback:</strong> {feedback}</p>
        </div>
      )}
    </div>
  );
};

export default ImageUpload;
