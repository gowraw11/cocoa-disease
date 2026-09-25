import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Paths to the saved models
leaf_model_path = 'cocoa_2.h5'  # Update this path to your cocoa fruit detection model
disease_model_path = 'cocoa_disease_model.h5'  # Update this path to your cocoa disease classification model

# Load the models
leaf_model = tf.keras.models.load_model(leaf_model_path)
disease_model = tf.keras.models.load_model(disease_model_path)

# Function to classify if the image is a cocoa fruit
def is_cocoa_fruit(img_path, model=leaf_model):
    # Adjust target size based on the model's expected input shape
    target_size = (leaf_model.input_shape[1], leaf_model.input_shape[2])
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    return prediction[0] > 0.5

# Function to classify the disease if the image is a cocoa fruit
def classify_disease(img_path, model=disease_model):
    # Adjust target size based on the model's expected input shape
    target_size = (disease_model.input_shape[1], disease_model.input_shape[2])
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    class_names = [
        'black_pod_rot', 'healthy', 'healthy_borer', 'pod_borer'
    ]
    predicted_class = class_names[np.argmax(prediction)]
    return predicted_class

# Main function to handle image classification
def process_image(img_path):
    if not is_cocoa_fruit(img_path):
        return "Invalid input. Not a cocoa fruit."
    else:
        disease_result = classify_disease(img_path)
        return "Cocoa fruit detected. Disease classification: {disease_result}"

# Example usage
img_path = '/Users/deekshithsy/Desktop/WhatsApp Image 2024-07-25 at 12.21.15.jpeg'  # Update this path to the image you want to test
result = process_image(img_path)
print(result)
