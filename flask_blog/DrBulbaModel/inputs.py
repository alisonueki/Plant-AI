import os
import cv2
from . import leaf_sick
import pandas as pd
import numpy as np

def test_image(model, image_path):
    image = cv2.imread(image_path)
    preprocessed_image = leaf_sick.preprocess_image(image)  # Preprocess the image

    # Calculate the mean values of R, G, and B channels
    mean_r, mean_g, mean_b = leaf_sick.calculate_mean(preprocessed_image)

    # Flatten the image into a 1D array
    flattened_image = preprocessed_image.flatten()

    # Use the trained model to predict on the image features
    prediction = model.predict([flattened_image])  # Assuming the model expects a 2D array

    return prediction