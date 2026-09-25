import os
import json
import numpy as np
from PIL import Image
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, 'class_indices.json'), 'r') as f:
    class_indices = json.load(f)

# Invert index mapping
index_to_class = {v: k for k, v in class_indices.items()}

# Load models
disease_model = tf.keras.models.load_model(os.path.join(BASE_DIR, 'cocoa_disease_model.keras'))
detector_model = tf.keras.models.load_model(os.path.join(BASE_DIR, 'cocoa_2.keras'))

def predict_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img_resized = img.resize((224, 224))
    img_arr = np.array(img_resized, dtype=np.float32) / 255.0
    img_batch = np.expand_dims(img_arr, axis=0)

    # 1. Cocoa Detection
    detector_prob = float(detector_model.predict(img_batch, verbose=0)[0][0])
    is_cocoa = detector_prob > 0.5

    # 2. Disease Classification
    disease_probs = disease_model.predict(img_batch, verbose=0)[0]
    top_class_idx = int(np.argmax(disease_probs))
    top_class_name = index_to_class[top_class_idx]
    confidence = float(disease_probs[top_class_idx])

    breakdown = {index_to_class[i]: round(float(disease_probs[i]) * 100, 2) for i in range(len(disease_probs))}

    return {
        'is_cocoa': is_cocoa,
        'cocoa_score': round(detector_prob * 100, 2),
        'top_class': top_class_name,
        'confidence': round(confidence * 100, 2),
        'probabilities': breakdown
    }

# Test on 1 sample from each disease class
for class_name in ['black_pod_rot', 'healthy', 'healthy_borer', 'pod_borer']:
    class_dir = os.path.join(BASE_DIR, 'image', class_name)
    sample_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if sample_files:
        test_file = os.path.join(class_dir, sample_files[0])
        res = predict_image(test_file)
        print(f"\n[Test Class: {class_name}] -> File: {sample_files[0]}")
        print(f"  Is Cocoa Fruit: {res['is_cocoa']} ({res['cocoa_score']}%)")
        print(f"  Predicted Disease: {res['top_class']} (Confidence: {res['confidence']}%)")
        print(f"  Class Breakdown: {res['probabilities']}")

# Test on 1 non-cocoa sample
non_cocoa_dir = os.path.join(BASE_DIR, 'dataset', 'train', 'non_cocoa')
non_cocoa_files = [f for f in os.listdir(non_cocoa_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
if non_cocoa_files:
    test_file = os.path.join(non_cocoa_dir, non_cocoa_files[0])
    res = predict_image(test_file)
    print(f"\n[Test Class: NON-COCOA] -> File: {non_cocoa_files[0]}")
    print(f"  Is Cocoa Fruit: {res['is_cocoa']} ({res['cocoa_score']}%)")
    print(f"  Detector correctly identified non-cocoa: {not res['is_cocoa']}")
