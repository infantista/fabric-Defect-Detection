import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def load_and_preprocess(data_dir, img_size=(128, 128)):
    """Load images from directory and preprocess them"""
    images = []
    labels = []
    
    for label, category in enumerate(['good', 'defective']):
        category_dir = os.path.join(data_dir, category)
        for img_name in os.listdir(category_dir):
            img_path = os.path.join(category_dir, img_name)
            try:
                img = cv2.imread(img_path)
                img = cv2.resize(img, img_size)
                img = img / 255.0 
                images.append(img)
                labels.append(label)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
    
    return np.array(images), np.array(labels)

def create_train_test_split(data_dir, test_size=0.2):
    """Create train-test split and save to separate folders"""
    X, y = load_and_preprocess(data_dir)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y)
    return X_train, X_test, y_train, y_test