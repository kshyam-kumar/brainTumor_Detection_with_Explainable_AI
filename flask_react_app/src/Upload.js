import React, { useState } from "react";
import axios from "axios";
import "./Upload.css"
const Upload = () => {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState("");
  const [imageName, setImageName] = useState("");
  const [loading, setLoading] = useState(false);  // Add loading state

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    const formData = new FormData();
    formData.append("file", file);

    setLoading(true);  // Start loading when form is submitted

    try {
      const response = await axios.post(
        "http://127.0.0.1:5000/predict",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );
      setResult(response.data.prediction);
      setImageName(response.data.image_name);
    } catch (error) {
      console.error("Error uploading file:", error);
    } finally {
      setLoading(false);  // Stop loading when the request completes
    }
  };

  const GradCamImagePath = `http://localhost:5000/gradcam/${imageName}`;
  const originalImagePath = `http://localhost:5000/original_image/${imageName}`;
  const saliencyImagePath = `http://localhost:5000/saliency_image/${imageName}`;
  const cannyImagePath = `http://localhost:5000/canny_image/${imageName}`;
  return (
    <div>
      <h1>Brain Tumor Classification</h1>
      <div className="container">
        <form onSubmit={handleSubmit} className="input-box">
          <input
            type="file"
            accept=".png, .jpg, .jpeg"
            onChange={handleFileChange}
          />
          <button type="submit">Predict</button>
        </form>
      </div>
      

      {/* Show loading spinner while processing */}
      {loading && (
        <div>
          <p className="process">Processing, please wait...</p>
          {/* You can replace this text with a spinner or loading animation */}
        </div>
      )}

      {/* Show the result and images after processing */}
      {!loading && result && (
        <div>
          <h3>Prediction: {result}</h3>
          <div className="image-container" style={{ display: "flex", gap: "20px" }}>
            <div>
              <h3>Original Image:</h3>
              <img src={originalImagePath} alt="Original" height="300px" width="300px" />
            </div>
            <div>
              <h3>GradCAM Result:</h3>
              <img src={GradCamImagePath} alt="GradCAM result" height="300px" width="300px" />
            </div>
            <div>
              <h3>Saliency Map Result:</h3>
              <img src={saliencyImagePath} alt="Saliency result" height="300px" width="300px" />
            </div>
            <div>
              <h3>Canny Result:</h3>
              <img src={cannyImagePath} alt="Canny result" height="300px" width="300px" />
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Upload;
