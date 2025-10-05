# src/data_processing.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import joblib

def load_data(path):
    """Loads data from a CSV file."""
    print("Loading data...")
    return pd.read_csv(path)

def create_features(df):
    """Performs feature engineering."""
    print("Creating new features...")
    df_processed = df.drop('customer_id', axis=1)
    df_processed['debt_to_income_ratio'] = df_processed['total_debt_outstanding'] / (df_processed['income'] + 1e-6)
    df_processed['loan_to_income_ratio'] = df_processed['loan_amt_outstanding'] / (df_processed['income'] + 1e-6)
    print("Feature engineering complete.")
    return df_processed

def cap_outliers(df, column, lower_bound, upper_bound):
    """Caps outliers in a specified column using pre-calculated bounds."""
    df[column] = np.clip(df[column], lower_bound, upper_bound)
    return df

def save_processed_data(output_path, **kwargs):
    """Saves the processed dataframes to CSV files."""
    print("Saving processed data...")
    os.makedirs(output_path, exist_ok=True)
    for name, df in kwargs.items():
        df.to_csv(os.path.join(output_path, f"{name}.csv"), index=False)
    print(f"Data saved to {output_path}")

def main():
    """Main function to run the data processing pipeline."""
    input_data_path = 'data/Loan_Data.csv'
    output_data_path = 'data/processed'
    
    # 1. Load, 2. Create Features, 3. Split Features/Target
    df = load_data(input_data_path)
    df_featured = create_features(df.copy())
    X = df_featured.drop('default', axis=1)
    y = df_featured['default']
    
    # 4. Split into training and testing sets
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 5. Handle outliers (learning bounds from training data)
    print("Handling outliers...")
    columns_to_cap = ['income', 'fico_score', 'total_debt_outstanding', 'loan_amt_outstanding']
    for col in columns_to_cap:
        Q1 = X_train[col].quantile(0.25)
        Q3 = X_train[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        X_train = cap_outliers(X_train.copy(), col, lower_bound, upper_bound)
        X_test = cap_outliers(X_test.copy(), col, lower_bound, upper_bound)
    
    # 6. Scale features (learning scaling parameters from training data)
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # --- ADD THIS SECTION TO SAVE THE SCALER ---
    # Save the fitted scaler object for later use in the app
    os.makedirs('processors', exist_ok=True)
    joblib.dump(scaler, 'processors/scaler.joblib')
    print("Scaler saved to processors/scaler.joblib")
    # --- END OF ADDED SECTION ---

    # Convert scaled arrays back to DataFrames for saving
    X_train = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test = pd.DataFrame(X_test_scaled, columns=X.columns)
    print("Feature scaling complete.")

    # 7. Save the final processed data
    save_processed_data(
        output_data_path,
        X_train=X_train,
        X_test=X_test,
        y_train=pd.DataFrame(y_train),
        y_test=pd.DataFrame(y_test)
    )

if __name__ == '__main__':
    main()