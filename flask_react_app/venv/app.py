import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model
from werkzeug.utils import secure_filename
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model
model = load_model('model/brain.keras')

def get_className(classNo):
    if classNo == 0:
        return "Glioma"
    elif classNo == 1:
        return "Meningioma"
    elif classNo == 2:
        return "No Tumor"
    elif classNo == 3:
        return "Pituitary"

def getResult(img_path):
    image = cv2.imread(img_path)
    if image is None:
        raise ValueError(f"Image not found or could not be loaded: {img_path}")
    img_resized = cv2.resize(image, (240, 240))
    img_array = img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    probabilities = model.predict(img_array)[0]
    return np.argmax(probabilities)

def VizGradCAM(model, image,fname, interpolant=0.5, plot_results=False):
    """ VizGradCAM - Displays GradCAM based on Keras / TensorFlow models
        using the gradients from the last convolutional layer. 
    """
    # Ensure interpolant is within the valid range
    img = cv2.imread(image)
    img = cv2.resize(img, (240, 240))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    assert (interpolant > 0 and interpolant < 1), "Heatmap Interpolation Must Be Between 0 - 1"

    # STEP 1: Preprocess image and make prediction using the model
    original_img = np.asarray(img, dtype=np.float32)
    img = np.expand_dims(original_img, axis=0)
    prediction = model.predict(img)
    prediction_idx = np.argmax(prediction)

    # STEP 2: Identify the last convolutional layer in the model
    last_conv_layer = next(x for x in model.layers[::-1] if isinstance(x, tf.keras.layers.Conv2D))
    target_layer = model.get_layer(last_conv_layer.name)

    with tf.GradientTape() as tape:
        gradient_model = Model([model.inputs], [target_layer.output, model.output])
        conv2d_out, prediction = gradient_model(img)
        loss = prediction[:, prediction_idx]

    gradients = tape.gradient(loss, conv2d_out)
    output = conv2d_out[0]

    # Depthwise mean to get the weights for the heatmap
    weights = tf.reduce_mean(gradients[0], axis=(0, 1))
    activation_map = np.zeros(output.shape[0:2], dtype=np.float32)
    
    # Create activation map
    for idx, weight in enumerate(weights):
        activation_map += weight * output[:, :, idx]
    
    activation_map = cv2.resize(activation_map.numpy(), (original_img.shape[1], original_img.shape[0]))
    activation_map = np.maximum(activation_map, 0)
    activation_map = (activation_map - activation_map.min()) / (activation_map.max() - activation_map.min())
    activation_map = np.uint8(255 * activation_map)

    # Convert to heatmap
    heatmap = cv2.applyColorMap(activation_map, cv2.COLORMAP_JET)

    # Superimpose heatmap on the original image
    original_img = np.uint8((original_img - original_img.min()) / (original_img.max() - original_img.min()) * 255)
    cvt_heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    cvt_heatmap = img_to_array(cvt_heatmap)
    blended_image = np.uint8(original_img * interpolant + cvt_heatmap * (1 - interpolant))
    
    # Save the blended image to the specified path
    
    
    plt.imsave(f"../venv/gradCamUpload/{fname}.jpg", blended_image)  
    plt.rcParams["figure.dpi"] = 100

    if plot_results:
        plt.imshow(np.uint8(original_img * interpolant + cvt_heatmap * (1 - interpolant)))
    else:
        return blended_image



def compute_saliency_map(model, image,fname):
    """ Computes the Saliency Map for a given input image using the specified model. """
    # Convert the image to a tensor
    test_img = cv2.imread(image)
    test_img = cv2.resize(test_img, (240, 240))
    test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
    test_img=img_to_array(test_img)
    image_tensor = tf.convert_to_tensor(test_img, dtype=tf.float32)
    image_tensor = tf.expand_dims(image_tensor, axis=0)  # Add batch dimension
    
    with tf.GradientTape() as tape:
        # Watch the input image
        tape.watch(image_tensor)
        # Get model prediction
        prediction = model(image_tensor)
        predicted_class = tf.argmax(prediction[0]).numpy()  # Get the class index

        # Calculate the gradient of the predicted class with respect to the input image
        grads = tape.gradient(prediction[0][predicted_class], image_tensor)

    # Get the saliency map by taking the maximum absolute gradient across the color channels
    saliency = tf.reduce_max(tf.abs(grads), axis=-1).numpy()[0]  # Get the saliency for the first (and only) image

    # Normalize the saliency map
    saliency = (saliency - np.min(saliency)) / (np.max(saliency) - np.min(saliency))
    
    # Scale the saliency map to enhance visibility (increase the factor for more brightness)
    saliency = np.clip(saliency * 5, 0, 1)  # Amplify bright areas even more
    saliency = np.uint8(255 * saliency)  # Convert to uint8 format
    
    file_path="../venv/SaliencyImages/"
    
    
    plt.imsave(f"../venv/SaliencyImages/{fname}.jpg", saliency,cmap='hot')
    
    plt.close()
    return saliency

def canny_algorithm(img_path, fname):
    # Load the image from the provided path
    image = cv2.imread(img_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

    # Save the Canny edge result
    cv2.imwrite(f"../venv/canny/{fname}.jpg", edges)

@app.route('/predict', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided.'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected.'}), 400

    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)
    value = getResult(file_path)
    fname = os.path.splitext(file.filename)[0]
    gradcam_result=VizGradCAM(model,file_path,fname,plot_results=False)
    saliency_result=compute_saliency_map(model,file_path,fname)
    canny_algorithm(file_path,fname)
    result = get_className(value)  # Fixed: Removed
    image_name = f"{fname}.jpg"

    #plot_results
    return jsonify({
        'prediction': result,
        'image_name': image_name
    })

@app.route('/gradcam/<fname>')
def serve_gradcam_image(fname):
    return send_from_directory("gradCamUpload", fname)

@app.route('/original_image/<fname>')
def serve_original_image(fname):
    return send_from_directory("uploads", fname)

@app.route('/saliency_image/<fname>')
def serve_saliency_image(fname):
    return send_from_directory("SaliencyImages", fname)

@app.route('/canny_image/<fname>')
def serve_canny_image(fname):
    return send_from_directory("canny", fname)

if __name__ == '__main__':
    app.run(debug=True)
