import numpy as np
import tensorflow as tf
import cv2
import os
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# Define class labels
class_labels = ['Biryani', 'Butter chicken', 'Chapati', 'Chicken tandoori', 'Chole bhature',
                    'Dal Makhani', 'Dhokla', 'Dosa', 'Gajar halwa', 'Ghevar',
                    'Gulab jamun', 'Idli', 'Jalebi', 'Kadai paneer', 'Kathi roll',
                    'Kofta', 'Masala bhindi', 'Medu vada', 'Pani puri', 'Pav bhaji',
                    'Poori', 'Rasgulla', 'Samosa', 'Toor dal', 'Vada pav']

# Cache for the model
_model = None

def get_model(model_path=None):
    """Lazy loading of the model"""
    global _model
    if _model is None:
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'dish_classification_model1.h5')
        
        try:
            _model = load_model(model_path)
        except Exception as e:
            raise Exception(f"Failed to load model: {str(e)}")
    return _model

def preprocess_image(image_path):
    """Preprocess image for model prediction"""
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise Exception("Failed to read image")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (128, 128))
        image = img_to_array(image) / 255.0
        image = np.expand_dims(image, axis=0)
        return image
    except Exception as e:
        raise Exception(f"Error preprocessing image: {str(e)}")

def predict_image(image_path):
    """Predict dish from image"""
    try:
        model = get_model()
        image = preprocess_image(image_path)
        predictions = model.predict(image)
        class_index = np.argmax(predictions)
        confidence = float(np.max(predictions))
        predicted_class = class_labels[class_index]
        return predicted_class, confidence
    except Exception as e:
        raise Exception(f"Prediction error: {str(e)}")