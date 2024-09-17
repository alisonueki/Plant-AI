import os
import cv2
import leaf_sick
import numpy as np

def process_user_image(model, image_path, mean_df):
    # Check if the image is in PNG format
    if not image_path.lower().endswith('.png'):
        print("Only PNG images are allowed.")
        return

    # Load image from path
    user_image = cv2.imread(image_path)
    # Preprocess user image
    preprocessed_user_image = leaf_sick.preprocess_image(user_image)

    # Calculate the mean values and standard deviations for the user image
    user_mean_r, user_mean_g, user_mean_b = leaf_sick.calculate_mean(preprocessed_user_image)
    user_std_r, user_std_g, user_std_b = preprocessed_user_image.std(axis=(0, 1))

    # Calculate the mean and standard deviation for the training dataset
    mean_r_mean = mean_df['Mean R'].mean()
    mean_g_mean = mean_df['Mean G'].mean()
    mean_b_mean = mean_df['Mean B'].mean()
    std_r_mean = mean_df['Std R'].mean()
    std_g_mean = mean_df['Std G'].mean()
    std_b_mean = mean_df['Std B'].mean()

    mean_r_std = mean_df['Mean R'].std()
    mean_g_std = mean_df['Mean G'].std()
    mean_b_std = mean_df['Mean B'].std()
    std_r_std = mean_df['Std R'].std()
    std_g_std = mean_df['Std G'].std()
    std_b_std = mean_df['Std B'].std()

    # Calculate the distance between user image and training dataset
    mahalanobis_distance = np.sqrt(
        ((user_mean_r - mean_r_mean) / mean_r_std) ** 2 +
        ((user_mean_g - mean_g_mean) / mean_g_std) ** 2 +
        ((user_mean_b - mean_b_mean) / mean_b_std) ** 2 +
        ((user_std_r - std_r_mean) / std_r_std) ** 2 +
        ((user_std_g - std_g_mean) / std_g_std) ** 2 +
        ((user_std_b - std_b_mean) / std_b_std) ** 2
    )

    # Define a threshold for distance to classify as sick or healthy
    # <= 3 is sick
    threshold = 3.0 

    if mahalanobis_distance > threshold:
        print("The image is healthy.")
        prediction = 'healthy'
    else:
        print("The image is sick.")
        prediction = 'sick'

    return mahalanobis_distance, prediction

def test_images(model, mean_df):
    folder_path_sick = '/Users/alisonueki/Desktop/plant/sick'
    folder_path_healthy = '/Users/alisonueki/Desktop/plant/healthy'

    correct_sick = 0
    correct_healthy = 0
    total_sick = 0
    total_healthy = 0

    # Lists to store confidence scores and predictions for all tested images
    confidence_scores = []
    predictions = []

    # Test images in the "sick" category
    print("Sick:")
    for filename in os.listdir(folder_path_sick):
        if filename.lower().endswith('.png'):
            total_sick += 1
            image_path = os.path.join(folder_path_sick, filename)
            mahalanobis_distance, prediction = process_user_image(model, image_path, mean_df)
            if mahalanobis_distance is not None and prediction is not None:
                if prediction == 'sick':
                    correct_sick += 1
                # Append confidence score and prediction to lists
                confidence_scores.append(mahalanobis_distance)
                predictions.append(prediction)

    # Test images in the "healthy" category
    print("Healthy:")
    for filename in os.listdir(folder_path_healthy):
        if filename.lower().endswith('.png'):
            total_healthy += 1
            image_path = os.path.join(folder_path_healthy, filename)
            mahalanobis_distance, prediction = process_user_image(model, image_path, mean_df)
            if mahalanobis_distance is not None and prediction is not None:
                if prediction == 'healthy':
                    correct_healthy += 1
                # Append confidence score and prediction to lists
                confidence_scores.append(mahalanobis_distance)
                predictions.append(prediction)

    # Calculate and report accuracy for each category
    accuracy_sick = (correct_sick / total_sick) * 100
    accuracy_healthy = (correct_healthy / total_healthy) * 100
    overall_accuracy = ((correct_sick + correct_healthy) / (total_sick + total_healthy)) * 100

    print(f"Accuracy for 'sick' category: {accuracy_sick:.2f}%")
    print(f"Accuracy for 'healthy' category: {accuracy_healthy:.2f}%")
    print(f"Overall Accuracy: {overall_accuracy:.2f}%")

