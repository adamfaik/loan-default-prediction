# src/train.py

import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import ParameterGrid
import lightgbm as lgb
import os
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

def load_processed_data(path):
    """Loads the pre-processed training and testing data."""
    X_train = pd.read_csv(os.path.join(path, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(path, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(path, "y_train.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join(path, "y_test.csv")).values.ravel()
    return X_train, X_test, y_train, y_test

def evaluate_model(model, X_test, y_test):
    """Evaluates the model and returns performance metrics."""
    predictions = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1_score": f1_score(y_test, predictions)
    }
    return metrics

def main():
    """Main function to run the training experiments."""
    processed_data_path = 'data/processed'
    
    # Define the models we want to test
    models = {
        "LogisticRegression": LogisticRegression(),
        "RandomForest": RandomForestClassifier(),
        "LightGBM": lgb.LGBMClassifier()
    }

    # Define the hyperparameter grids for each model
    hyperparameters = {
        "LogisticRegression": {
            "solver": ["liblinear"],
            "C": [0.1, 1.0, 10],
            "random_state": [42]
        },
        "RandomForest": {
            "n_estimators": [50, 100],
            "max_depth": [5, 10],
            "random_state": [42]
        },
        "LightGBM": {
            "n_estimators": [50, 100],
            "learning_rate": [0.05, 0.1],
            "random_state": [42]
        }
    }

    # Load data once
    X_train, X_test, y_train, y_test = load_processed_data(processed_data_path)

    # Loop through each model
    for model_name, model_instance in models.items():
        print(f"--- Training {model_name} ---")
        mlflow.set_experiment(f"{model_name}_Experiment")
        
        param_grid = ParameterGrid(hyperparameters[model_name])
        
        # Loop through each hyperparameter combination
        for params in param_grid:
            with mlflow.start_run():
                print(f"Running with params: {params}")
                
                # Log hyperparameters
                mlflow.log_params(params)
                
                # Train model
                model = model_instance.set_params(**params)
                model.fit(X_train, y_train)
                
                # Evaluate model
                metrics = evaluate_model(model, X_test, y_test)
                print(f"Metrics: {metrics}")
                
                # Log metrics
                mlflow.log_metrics(metrics)
                
                # Log model
                mlflow.sklearn.log_model(model, f"{model_name}_model")

    print("\nAll experiments complete. Check the MLflow UI.")

if __name__ == '__main__':
    main()