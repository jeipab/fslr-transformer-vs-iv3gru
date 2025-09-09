import tensorflow as tf

from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
import cv2
import numpy as np

# Load the InceptionV3 model pre-trained on ImageNet
iv3_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')

def extract_iv3_features(frame_bgr, image_size=(299, 299)):
     """
     Extracts IV3 features for a given frame.
     :param frame_bgr: Input frame in BGR format (OpenCV).
     :param image_size: Size to resize the image for InceptionV3 input (default is 299x299).
     :return: IV3 feature vector (2048,)
     """
     # Convert BGR to RGB
     frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

     # Resize image to 299x299 (required input size for InceptionV3)
     img_resized = cv2.resize(frame_rgb, image_size)

     # Preprocess image for InceptionV3
     img_array = image.img_to_array(img_resized)
     img_array = np.expand_dims(img_array, axis=0)
     img_array = preprocess_input(img_array)  # ImageNet normalization

     # Extract features using InceptionV3 model
     features = iv3_model.predict(img_array)
     return features.flatten()  # Flatten to get the 2048-dimensional vector
