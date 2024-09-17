import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

total = 0
mean_array = []
data_dict = {
    'Classification': [],
    'Mean R': [],
    'Mean G': [],
    'Mean B': [],
    'Std R': [],
    'Std G': [],
    'Std B': []
}

def get_data_set(image_path):
    if 'healthy' in image_path:
        data_dict['Classification'].append('healthy')
    else:
        data_dict['Classification'].append('sick')
        
    global total
    image = cv2.imread(image_path)
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    mean_rgb = image_rgb.mean(axis=(0, 1))
    std_rgb = image_rgb.std(axis=(0, 1))
    
    mean_array.append(mean_rgb)
    data_dict['Mean R'].append(mean_rgb[0])
    data_dict['Mean G'].append(mean_rgb[1])
    data_dict['Mean B'].append(mean_rgb[2])
    total += mean_rgb

    # Append standard deviations to the dictionary
    data_dict['Std R'].append(std_rgb[0])
    data_dict['Std G'].append(std_rgb[1])
    data_dict['Std B'].append(std_rgb[2])

def calculate_mean(image):
    # Calculate the mean values of R, G, and B channels for the image
    mean_r = np.mean(image[:, :, 0])
    mean_g = np.mean(image[:, :, 1])
    mean_b = np.mean(image[:, :, 2])
    return mean_r, mean_g, mean_b

def preprocess_image(image, desired_width=50, desired_height=50):
    # Resize the image to a fixed size
    image = cv2.resize(image, (desired_width, desired_height))

    # Convert to RGB color space
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image_rgb


def load(folder_path_sick, folder_path_healthy, test_size=0.2, random_state=42):
    X = []
    y = []

    # Load and preprocess sick plant images
    for folder_path, label in [(folder_path_sick, 'sick'), (folder_path_healthy, 'healthy')]:
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)

            if os.path.isfile(file_path):
                # Load and preprocess the image
                image = cv2.imread(file_path)
                preprocessed_image = preprocess_image(image)

                # Calculate the mean values of R, G, and B channels
                mean_r, mean_g, mean_b = calculate_mean(preprocessed_image)

                # Flatten the image into a 1D array
                flattened_image = preprocessed_image.flatten()

                X.append(flattened_image)
                y.append(label)

                # Call get_data_set to add to the total and data_dict
                get_data_set(file_path)

    # Convert mean values to a DataFrame
    mean_df = pd.DataFrame(data_dict)

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test), mean_df

def get_mean_array():
    return mean_array

def get_total_rgb():
    return total

def get_dict():
    return data_dict
