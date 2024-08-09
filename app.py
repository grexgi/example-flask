from flask import Flask, request, jsonify
import tensorflow as tf
# from tensorflow.python.keras.applications import EfficientNetB3
from tensorflow.python.keras.layers import Dense, GlobalAveragePooling2D, Dropout
import numpy as np
from PIL import Image
import io
from keras import preprocessing
from tensorflow.python.keras.models import Model

app = Flask(__name__)

def build_model():
    base_model = tf.keras.applications.EfficientNetB3(weights=None, include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(.5)(x)
    x = tf.keras.layers.Dense(4, activation='softmax')(x)  # Adjust the number of classes
    model = tf.keras.Model(inputs=base_model.input, outputs=x)
    return model

# Load weights into the model
model = build_model()
model_path = './model/EfficientNetB3_iterasike3-36.weights.h5'
# Load your model (change the path to your model file if necessary)
model.load_weights(model_path)

# Define the mapping from integers to labels
label_map = {
    0: 'Dead Plants',
    1: 'Moderately Healthy Plants',
    2: 'Unhealthy Plants',
    3: 'Very Healthy Plants'
}

def preprocess_image(img):
    """Preprocess the image to match the input expected by your model."""
    img = img.resize((224, 224))  # Adjust size based on your model's expected input
    img_array = np.array(img)
    img_array = img_array / 255.0  # Normalization if your model expects it
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/')
def hello_world():
    return 'Welcome to crop health classification service (^-^)/'

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        img = Image.open(io.BytesIO(file.read()))
        img_array = preprocess_image(img)
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        label = label_map.get(predicted_class, 'Unknown')
        return jsonify({'prediction': label}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
