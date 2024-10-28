import logging
import os
import time
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
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

# Path to the training data directory (if needed for data generators)
train_data_path = os.path.join(os.getcwd(), "Training_data")

# Initialize and create a train data generator if needed (optional)
try:
    datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    train_generator = datagen.flow_from_directory(
        train_data_path,
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary'
    )
    logging.info("Training data generator created successfully.")
except FileNotFoundError as e:
    logging.error(f"Training data directory not found: {e}")

# Load the model with a relative path
model_path = "goat_classifier_simple_cnn.h5"  # Model file name in the project root
try:
    model = load_model(model_path)
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    model = None

# Preprocess image function
def preprocess_image(image_path):
    try:
        # Load and preprocess the image as per model requirements
        image = Image.open(image_path)
        image = image.resize((224, 224))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
        logging.info("Image preprocessed successfully.")
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
