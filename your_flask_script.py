import logging
import os
import time
from flask import Flask, request, jsonify
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import numpy as np

# Suppress TensorFlow warnings for cleaner logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Initialize Flask app
app = Flask(__name__)

# Set up logging to capture all details for debugging
logging.basicConfig(level=logging.INFO)

# Define a directory to save uploaded images
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Model path
model_path = "goat_classifier_simple_cnn.h5"

# Load or redefine model architecture if needed
try:
    # Load model if it exists
    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        # Define a basic CNN model with Flatten layer before dense layers
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Flatten(),  # Ensure Flatten layer is here
            Dense(128, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.save(model_path)  # Save the newly defined model for future use

    logging.info("Model loaded or created successfully.")
except Exception as e:
    logging.error(f"Error loading or creating model: {e}")
    model = None

# Preprocess image function
def preprocess_image(image_path):
    try:
        # Load and convert image to RGB
        image = Image.open(image_path).convert('RGB')
        
        # Resize the image to (224, 224) as expected by the model
        image = image.resize((224, 224))
        
        # Convert the image to a numpy array and normalize
        image_array = np.array(image) / 255.0
        
        # Ensure the image shape matches (224, 224, 3)
        image_array = np.reshape(image_array, (224, 224, 3))
        
        # Expand dimensions to match (1, 224, 224, 3) for batch processing
        image_array = np.expand_dims(image_array, axis=0)
        
        logging.info("Image preprocessed successfully with shape: %s", image_array.shape)
        return image_array
    except Exception as e:
        logging.error(f"Error in image preprocessing: {e}")
        raise

# Test route to confirm the server is running
@app.route('/test', methods=['GET'])
def test():
    return "Test route is working!"

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded successfully"}), 500

    try:
        logging.info("Received a request on /predict")
        
        # Save the uploaded image
        image_file = request.files['file']
        image_path = os.path.join(UPLOAD_FOLDER, f"{int(time.time())}_{image_file.filename}")
        image_file.save(image_path)
        logging.info(f"Image saved at: {image_path}")
        
        # Preprocess the image
        preprocessed_image = preprocess_image(image_path)
        
        # Make a prediction
        prediction = model.predict(preprocessed_image)
        logging.info(f"Prediction completed: {prediction}")

        return jsonify({'prediction': prediction.tolist()})
    
    except Exception as e:
        logging.error(f"An error occurred during prediction: {e}")
        return jsonify({'error': str(e)}), 500

# Run the app with dynamic port configuration for Render compatibility
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get('PORT', 5000)))
