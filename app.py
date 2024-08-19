import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2gray
import tensorflow as tf
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Define the Flask application
app = Flask(__name__)
CORS(app)  # Enable CORS
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

class DiseaseModel:
    def __init__(self):
        self.model = None
        self.label_map = {0: 'Daun Sehat', 1: 'Keriting Mozaik', 2: 'Virus Gemini'}
        self.input_shape = (100, 100, 1)

        try:
            self.model = load_model('./model/cnn_model_3.h5')
        except ValueError as e:
            print(f"Error loading model 1: {e}")

    def predict(self, img):
        if self.model is not None:
            img = resize(img, (100, 100), anti_aliasing=True)
            img = rgb2gray(img)  # Convert to grayscale
            img = img.reshape(1, 100, 100, 1)  # Reshape to match input shape of the model
            prediction = self.model.predict(img)
            predicted_class = np.argmax(prediction, axis=1)[0]
            return self.label_map[predicted_class]
        else:
            return "Model 1 not loaded"

class VegetationModel:

    def __init__(self):
        self.model = None
        self.label_map = {0: 'Dead Plants', 1: 'Moderately Healthy Plants', 2: 'Unhealthy Plants', 3: 'Very Healthy Plants'}
        self.input_shape = (224, 224, 3)

        def build_model():
            base_model = tf.keras.applications.EfficientNetB3(weights=None, include_top=False, input_shape=(224, 224, 3))
            x = base_model.output
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            x = tf.keras.layers.Dense(128, activation='relu')(x)
            x = tf.keras.layers.Dropout(.5)(x)
            x = tf.keras.layers.Dense(4, activation='softmax')(x)  # Adjust the number of classes
            model = tf.keras.Model(inputs=base_model.input, outputs=x)
            return model
        
        try:
            self.model = build_model()
            self.model.load_weights('./model/EfficientNetB3_iterasike3-36.weights.h5')
        except ValueError as e:
            print(f"Error loading model 2: {e}")

    def predict(self, img):
        if self.model is not None:
            img = resize(img, (224, 224), anti_aliasing=True)
            img = img.reshape(1, 224, 224, 3)  # Reshape to match input shape of the model
            prediction = self.model.predict(img)
            predicted_class = np.argmax(prediction, axis=1)[0]
            return self.label_map[predicted_class]
        else:
            return "Model 2 not loaded"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

DiseaseModel = DiseaseModel()
VegetationModel = VegetationModel()

@app.route('/predict_disease', methods=['POST'])
def predict_disease():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        
        try:
            img = imread(file)
            class_label = DiseaseModel.predict(img)
        except Exception as e:
            return jsonify({"error": f"Error processing image: {e}"}), 500

        return jsonify({"class_label": class_label}), 200
    else:
        return jsonify({"error": "File type not allowed"}), 400

@app.route('/predict_vegetation_class', methods=['POST'])
def predict_vegetation_class():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        
        try:
            img = imread(file)
            class_label = VegetationModel.predict(img)
        except Exception as e:
            return jsonify({"error": f"Error processing image: {e}"}), 500

        return jsonify({"class_label": class_label}), 200
    else:
        return jsonify({"error": "File type not allowed"}), 400

@app.route('/', methods=['GET'])
def home():
    return jsonify({"welcome": "Welcome to crop disease and ndvi prediction model"}),200

if __name__ == '__main__':
    app.run(debug=True)