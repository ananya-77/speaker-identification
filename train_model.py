import os
import librosa
import numpy as np
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
import json

# Define constants
SPEAKER_DATA_FOLDER = "C:/Users/Ananya/Desktop/speaker-identification/16000_pcm_speeches"  # Your data folder
MODEL_SAVE_PATH = "C:/Users/Ananya/Desktop/speaker-identification/speaker_identification_model.h5"
LABEL_ENCODER_SAVE_PATH = "C:/Users/Ananya/Desktop/speaker-identification/label_encoder.json"

# Helper function to preprocess audio files
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

# Prepare the dataset
X, y = [], []
speaker_dirs = [os.path.join(SPEAKER_DATA_FOLDER, speaker) for speaker in os.listdir(SPEAKER_DATA_FOLDER)]

# Ensure we only process directories (not files like tf_Wav_reader.py)
speaker_dirs = [dir for dir in speaker_dirs if os.path.isdir(dir)]

# Loop through each speaker directory
for speaker_dir in speaker_dirs:
    speaker_label = os.path.basename(speaker_dir)
    
    # Only process files inside the speaker directory
    for file in os.listdir(speaker_dir):
        file_path = os.path.join(speaker_dir, file)

        # Check if the path is a valid file (skip non-file items like scripts)
        if os.path.isfile(file_path) and file.lower().endswith(('.wav', '.mp3', '.flac')):  # You can specify valid audio file formats here
            features = preprocess_audio(file_path)
            if features is not None:
                X.append(features)
                y.append(speaker_label)

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Save the label encoder
with open(LABEL_ENCODER_SAVE_PATH, "w") as file:
    json.dump(label_encoder.classes_.tolist(), file)

# Build the neural network model
model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X.shape[1],)),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(label_encoder.classes_), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X, y_encoded, epochs=10, batch_size=32)

# Save the model
model.save(MODEL_SAVE_PATH)

print("Model and label encoder saved successfully!")
