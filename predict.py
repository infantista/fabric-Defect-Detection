import cv2
import numpy as np
import tensorflow as tf
import os

class FabricDefectDetector:
    def __init__(self, model_path):
        """Initialize with explicit model path"""
        self.model_path = os.path.abspath(model_path)
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at: {self.model_path}")
        
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            self.img_size = (128, 128)
            print(f"Successfully loaded model from {self.model_path}")
        except Exception as e:
            raise ValueError(f"Error loading model: {str(e)}")

    def preprocess(self, img_path):
        """Preprocess an image for prediction"""
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not read image: {img_path}")
        img = cv2.resize(img, self.img_size)
        img = img / 255.0  
        return np.expand_dims(img, axis=0)

    def predict(self, img_path):
        """Make a prediction on an image"""
        try:
            processed_img = self.preprocess(img_path)
            prediction = self.model.predict(processed_img)
            confidence = float(prediction[0][0] if prediction[0].shape == (1,) else float(prediction[0]))
            
            return {
                'status': 'Defective' if prediction[0] > 0.5 else 'Good',
                'confidence': confidence,
                'error': None
            }
        except Exception as e:
            return {
                'status': None,
                'confidence': 0.0,
                'error': str(e)
            }

# For testing directly
if __name__ == '__main__':
    detector = FabricDefectDetector(model_path="models/fabric_defect.h5")
    test_img = "data/test/defective/sample1.jpg"  #test image
    result = detector.predict(test_img)
    print(f"Prediction: {result}")