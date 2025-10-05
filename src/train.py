# src/train.py

import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

def load_processed_data(path):
    """Loads the pre-processed training and testing data."""
    X_train = pd.read_csv(os.path.join(path, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(path, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(path, "y_train.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join(path, "y_test.csv")).values.ravel()
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, hyperparameters):
    """Trains a Logistic Regression model with given hyperparameters."""
    # Initialize the model with the specified hyperparameters
    model = LogisticRegression(**hyperparameters)
    
    # Train the model
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluates the model and returns performance metrics."""
    # Make predictions
    predictions = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }
    return metrics

def main():
    """Main function to run the training experiment."""
    processed_data_path = 'data/processed'
    
    # Set up MLflow
    # This will create an experiment named "Loan_Default_Prediction" if it doesn't exist
    mlflow.set_experiment("Loan_Default_Prediction")
    
    # Define the model and hyperparameters for this run
    hyperparameters = {
        "solver": "liblinear",
        "random_state": 42
    }

    # Start an MLflow run
    with mlflow.start_run():
        print("Starting MLflow run...")
        
        # Log the hyperparameters
        mlflow.log_params(hyperparameters)
        print("Logged hyperparameters.")
        
        # Load data
        X_train, X_test, y_train, y_test = load_processed_data(processed_data_path)
        
        # Train the model
        print("Training model...")
        model = train_model(X_train, y_train, hyperparameters)
        print("Model training complete.")
        
        # Evaluate the model
        print("Evaluating model...")
        metrics = evaluate_model(model, X_test, y_test)
        print("Model evaluation complete.")
        print(f"Metrics: {metrics}")
        
        # Log the metrics
        mlflow.log_metrics(metrics)
        print("Logged metrics.")
        
        # Log the model itself
        mlflow.sklearn.log_model(model, "logistic_regression_model")
        print("Logged model artifact.")
        
        print("\nMLflow run complete. Check the UI at http://127.0.0.1:5000")

if __name__ == '__main__':
    main()