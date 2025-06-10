import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
# Load model and tokenizer
model_path = "bert_glaucoma_model"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()

# Define label mapping manually (based on original training)
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(["0", "1", "2"]) # original diagnosis values as strings

# Optional: map class IDs to names
label_names = {
    "0": "Healthy",
    "1": "Glaucoma Suspect",
    "2": "Glaucoma"
}

# ----- CUSTOM PATIENT INPUT -----
# Modify these values to test different patients
patient_data = {
    "age": 29,
    "gender": 0,  # 0 for male, 1 for female
    "dioptre_1": -1.75,
    "dioptre_2": -0.5,
    "astigmatism": 35,
    "phakic/pseudophakic": "Phakic",
    "pneumatic": 14.0,
    "perkins": 17.26,
    "pachymetry": 483,
    "axial_length": 24.39,
    "vf_md": -4.48
}

# Convert to input text
input_text = (
    f"Age: {patient_data['age']}, Gender: {'Male' if patient_data['gender'] == 0 else 'Female'}, "
    f"Dioptre 1: {patient_data['dioptre_1']}, Dioptre 2: {patient_data['dioptre_2']}, "
    f"Astigmatism: {patient_data['astigmatism']}, Phakic: {patient_data['phakic/pseudophakic']}, "
    f"IOP Pneumatic: {patient_data['pneumatic']}, Perkins: {patient_data['perkins']}, "
    f"Pachymetry: {patient_data['pachymetry']}, Axial Length: {patient_data['axial_length']}, "
    f"VF_MD: {patient_data['vf_md']}"
)

# Tokenize and predict
inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
with torch.no_grad():
    outputs = model(**inputs)
    predicted_class_id = torch.argmax(outputs.logits, dim=1).item()

# Decode label
predicted_label = label_encoder.inverse_transform([predicted_class_id])[0]
diagnosis_name = label_names[predicted_label]

print(f"Input text:\n{input_text}\n")
print(f"Predicted class ID: {predicted_class_id}")
print(f"Predicted diagnosis: {diagnosis_name}")
