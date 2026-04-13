from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

app = Flask(__name__)

# Define the path to your trained model and class labels (adjust if needed)
MODEL_PATH = r'C:\Users\Hp VICTUS\Downloads\fom\sduml\lung_cnn_pclf\vgg16_lung_model.h5'  # Use raw string for Windows paths
CLASS_LABELS = ['adenocarcinoma', 'large.cell.carcinoma', 'normal', 'squamous.cell.carcinoma']
IMAGE_SIZE = (224, 224)

# Load the model globally when the app starts
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None  # Important: Set model to None if loading fails

def preprocess_image(image_path):
    """Loads and preprocesses an image for prediction."""
    try:
        img = load_img(image_path, target_size=IMAGE_SIZE)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")  # Add error logging
        return None  # Return None on error

def predict(image_path):
    """Makes a prediction on the given image."""
    if model is None:
        return "Model not loaded."
    try:
        processed_image = preprocess_image(image_path)
        if processed_image is None:
            return "Image preprocessing failed."  # Handle preprocessing failure
        prediction = model.predict(processed_image)
        predicted_class_index = np.argmax(prediction)
        predicted_label = CLASS_LABELS[predicted_class_index]
        return predicted_label
    except Exception as e:
        return f"Prediction error: {e}"

@app.route('/', methods=['GET'])
def index():
    """Serves the main HTML page."""
    return render_template('index.html')  # Assumes you have an index.html in a 'templates' folder

@app.route('/predict', methods=['POST'])
def upload_and_predict():
    """Handles image upload and prediction."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided.'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No image selected.'}), 400

    if file:
        filename = secure_filename(file.filename)
        # Save the uploaded image temporarily.  Use a more robust temp directory.
        temp_filepath = os.path.join(app.root_path, 'tmp', filename) # saves in a /tmp subfolder
        os.makedirs(os.path.dirname(temp_filepath), exist_ok=True) #make the directory if it does not exist.
        try:
            file.save(temp_filepath)
            print(f"Image saved to: {temp_filepath}")

            # Make the prediction
            prediction_result = predict(temp_filepath)

            # Clean up the temporary file
            os.remove(temp_filepath)
            print(f"Temporary image removed: {temp_filepath}")
            return prediction_result, 200
        except Exception as e:
            return jsonify({'error': f'Error saving/processing file: {e}'}), 500
    else:
        return jsonify({'error': 'Invalid file format.'}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) #Explicitly set the port

