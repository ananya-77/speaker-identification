from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import librosa
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import json

# Initialize Flask App
app = Flask(__name__)

# Define paths for necessary files
MODEL_PATH = r"C:\Users\Ananya\Desktop\speaker-identification\speaker_identification_model.h5"
LABEL_ENCODER_PATH = r"C:\Users\Ananya\Desktop\speaker-identification\label_encoder.json"
UPLOAD_FOLDER = r"C:\Users\Ananya\Desktop\speaker-identification\uploads"  # Define the path for uploaded files

# Configure Flask upload folder
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained model
model = load_model(MODEL_PATH)

# Ensure the 'uploads' folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the label encoder from the saved JSON file
with open(LABEL_ENCODER_PATH, "r") as file:
    label_encoder_classes = json.load(file)
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(label_encoder_classes)

# Preprocess Audio File
def preprocess_audio(file_path, sr=16000, n_mfcc=13):
    try:
        audio, _ = librosa.load(file_path, sr=sr)
        audio, _ = librosa.effects.trim(audio)
        audio = librosa.util.normalize(audio)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Error processing audio: {e}")
        return None

# API to Predict Speaker
@app.route('/predict', methods=['POST'])
def predict_speaker():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Save uploaded file
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Preprocess and Predict
    features = preprocess_audio(file_path)
    if features is None:
        return jsonify({"error": "Audio processing failed"}), 500

    features = np.expand_dims(features, axis=0)
    predictions = model.predict(features)
    predicted_label = label_encoder.inverse_transform([np.argmax(predictions)])

    # Clean up the uploaded file
    os.remove(file_path)  

    return jsonify({"speaker": predicted_label[0]})

# Serve Frontend (HTML Template)
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
