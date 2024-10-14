Here's a detailed README file for your **Brain Tumor Detection with Explainable AI** project:

---

# Brain Tumor Detection with Explainable AI

## Overview

This project implements a brain tumor detection system using Convolutional Neural Networks (CNNs) with normalization and regularization techniques. The model, built using an **EfficientNet** architecture, achieves an impressive 98% accuracy in classifying brain tumors into four categories:
- Glioma
- Meningioma
- Pituitary Tumor
- No Tumor

Post-prediction, the system provides visual explanations through various techniques, helping users understand how the model makes predictions:
- **GradCAM** for identifying tumor regions.
- **Saliency Maps** to highlight important pixel areas.
- **Canny Edge Detection** to outline the tumor's edges.

## Features

- **EfficientNet-based CNN model** for high accuracy in tumor classification.
- 98% accuracy on test data.
- **Visual explainability** methods to make predictions interpretable:
  - **GradCAM**: Highlights the tumor region used by the model.
  - **Saliency Maps**: Displays pixel regions critical to the model’s decision.
  - **Canny Edge Detection**: Outlines the tumor edges for better visualization.

## Technologies Used

- **Python**
- **Jupyter Notebooks**
- **TensorFlow**
- **Keras**
- **OpenCV**
- **Scikit-Learn**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **React** for the frontend
- **Flask** for backend APIs

## Dataset

The dataset used in this project is sourced from Kaggle: [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset).

It is a combination of three datasets:
1. **Figshare**
2. **SARTAJ dataset**
3. **Br35H** (No tumor class images were taken from here)

The dataset contains **7023 images** of human brain MRI scans, classified into four types: **Glioma**, **Meningioma**, **Pituitary**, and **No Tumor**.

## Installation Instructions

### Clone the Repository

```bash
git clone https://github.com/kshyam-kumar/brainTumor_Detection_with_Explainable_AI
```

### Frontend Setup (React)

1. Navigate to the frontend directory:
   ```bash
   cd Gradcam/flask-react-app/frontend
   ```

2. Install the necessary dependencies:
   ```bash
   npm install react
   ```

3. Start the React development server:
   ```bash
   npm start
   ```

### Backend Setup (Flask)

1. Install the necessary Python packages:
   ```bash
   pip install flask tensorflow keras opencv-python pandas numpy scikit-learn matplotlib
   ```

2. Start the Flask server to handle model predictions and visual explainability:
   ```bash
   cd Gradcam/flask-react-app/backend
   flask run
   ```

## Usage

### Frontend (React):
The React-based user interface allows users to upload MRI images for prediction.

### Backend (Flask):
Flask handles the following tasks:
- Takes an MRI image as input.
- Runs the **EfficientNet** model to classify the tumor type.
- Generates visual explanations using **GradCAM**, **Saliency Maps**, and **Canny Edge Detection**.
- Saves and sends the visualized outputs back to the frontend.

## Model Architecture

1. **CNN Model**:
   - A custom convolutional neural network with multiple convolutional and dense layers.
   - Includes normalization and regularization techniques to improve generalization.

2. **EfficientNet Model**:
   - This model is pre-trained and fine-tuned for brain tumor classification.
   - It balances accuracy and efficiency by scaling depth, width, and resolution.

## Explainability Methods

This project implements three powerful explainability techniques:

1. **GradCAM (Gradient-weighted Class Activation Mapping)**:
   - Highlights the tumor region that influenced the model’s decision.

2. **Saliency Maps**:
   - Displays the pixel areas of the image that were critical in determining the model's output.

3. **Canny Edge Detection**:
   - Detects and outlines the edges of the tumor, making the model's prediction more interpretable.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Contributors

- [Your Name]

## Acknowledgments

- Thanks to the authors of the Kaggle dataset and the creators of the Figshare, SARTAJ, and Br35H datasets.

---

This README provides a clear structure for your project, making it easier for users to understand its features, installation process, and model architecture. You can adjust the content to suit any additional details you want to provide.
