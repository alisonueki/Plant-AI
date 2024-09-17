import os
import cv2
import joblib
from . import inputs
from . import leaf_sick
from . import model_generate
import pandas as pd
import numpy as np

'''
Accuracy for 'sick' category: 93.33%
Accuracy for 'healthy' category: 0.00%
Overall Accuracy: 46.67%
---------------------------------------
Accuracy for 'sick' category: 53.33%
Accuracy for 'healthy' category: 60.00%
Overall Accuracy: 56.67%
'''

def extract_data():
    total_mean = leaf_sick.get_total_rgb()/len(leaf_sick.get_mean_array())
    print("THIS IS THE TOTAL MEAN OF IT ALL: ", total_mean)
    global df
    df = pd.DataFrame(leaf_sick.get_dict())
    print(df)

def main():
    # Specify the paths to the datasets
    folder_path_sick = 'plant_sick'
    folder_path_healthy = 'plant_healthy'


    # Load and preprocess the data using the leaf_sick module
    X_train, y_train, X_test, y_test, mean_df = leaf_sick.load(folder_path_sick, folder_path_healthy)

    # Specify the path where you want to save the model
    model_save_path = 'Gabe_model'

    try:
        # Try to load the trained model
        model = joblib.load(model_save_path)
        print("Trained model loaded successfully.")
    except FileNotFoundError:
        # If the model file doesn't exist, train a new model
        model = model_generate.generate_model(X_train, y_train)
        # Save the trained model
        joblib.dump(model, model_save_path)
        print("Trained model and saved it to", model_save_path)

    # Evaluate the model on the test data
    #test_accuracy = model_generate.evaluate_model(model, X_test, y_test)
    #extract_data()    

    # Print the test accuracy
    #print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    #print(inputs.test_image(model, "sick.JPG"))

    # Process a user image
    #while True:
     #   user_image_path = input("Enter the path to the user image: ")
      #  if user_image_path.lower() == 'exit':
      #      break
      #  mahalanobis_distance, prediction = inputs.process_user_image(model, user_image_path, mean_df)
      #  print(f"Distance: {mahalanobis_distance:.2f}")
       # print(f"Prediction: {prediction}")

if __name__ == "__main__":
    main()
