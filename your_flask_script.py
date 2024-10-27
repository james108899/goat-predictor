import os
import logging
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from datetime import datetime
from PIL import UnidentifiedImageError

# Disable GPU usage (forces TensorFlow to use CPU only)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Initialize logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# Path to the saved model in H5 format
model_path = "goat_classifier_simple_cnn.h5"
try:
    model = load_model(model_path)
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")

# Directory to save uploaded images
upload_folder = "uploads"
os.makedirs(upload_folder, exist_ok=True)

# Test route to confirm the server is running
@app.route('/test', methods=['GET'])
def test():
    return "Test route is working!"

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    logging.info("Received a request on /predict")
    
    # Check if an image file is uploaded
    if 'image' not in request.files:
        logging.error("No image uploaded in request.")
        return jsonify({"error": "No image uploaded"}), 400
    
    image = request.files['image']

    try:
        # Save the uploaded image with a timestamp in the filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_path = os.path.join(upload_folder, f"{timestamp}_{image.filename}")
        image.save(img_path)
        logging.info(f"Image saved to {img_path}")

        # Load and preprocess the image
        img = load_img(img_path, target_size=(128, 128))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        logging.info("Image preprocessed successfully.")

        # Make prediction
        prediction = model.predict(img_array)[0][0]
        logging.info(f"Prediction made successfully: {prediction}")
        
        result = {
            'prediction': 'Mature Billy' if prediction > 0.5 else 'Nanny, Juvenile',
            'certainty': f"{prediction * 100:.2f}%"
        }
        
        # Save prediction results alongside the image
        with open(f"{img_path}_prediction.txt", "w") as f:
            f.write(str(result))
        
        return jsonify(result)

    except UnidentifiedImageError:
        logging.error("Uploaded file is not a valid image.")
        return jsonify({"error": "Uploaded file is not a valid image"}), 400
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return jsonify({"error": "An internal error occurred."}), 500

if __name__ == "__main__":
    app.run(debug=True)
