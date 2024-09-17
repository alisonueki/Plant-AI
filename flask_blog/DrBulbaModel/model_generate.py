import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def generate_model(X_train, y_train, n_estimators=100, random_state=42, model_save_path="trained_model"):
    # Create and configure a Random Forest classifier
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    
    # Train the model
    model.fit(X_train, y_train)

    # Save the trained model to a file
    joblib.dump(model, model_save_path)

    return model

def evaluate_model(model, X_test, y_test):
    # Make predictions on the test data
    y_test_pred = model.predict(X_test)

    # Calculate and return the test accuracy
    test_accuracy = accuracy_score(y_test, y_test_pred)

    return test_accuracy

