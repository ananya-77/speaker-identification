import json
from sklearn.preprocessing import LabelEncoder

# Assuming label_encoder has been fitted already
label_encoder = LabelEncoder()
label_encoder.fit(["Speaker_1", "Speaker_2", "Speaker_3"])  # Example fit, replace with your actual labels

# Save the LabelEncoder's classes to a json file
with open("label_encoder.json", "w") as file:
    json.dump(label_encoder.classes_.tolist(), file)

print("Label Encoder classes saved successfully!")
