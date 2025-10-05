# src/data_processing.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def load_data(path):
    """Loads data from a CSV file."""
    print("Loading data...")
    return pd.read_csv(path)

def create_features(df):
    """Performs feature engineering."""
    print("Creating new features...")
    
    # Drop the customer_id as it's not a predictive feature
    df_processed = df.drop('customer_id', axis=1)
    
    # Create Debt-to-Income Ratio
    # We add a small epsilon (1e-6) to income to avoid division by zero errors.
    df_processed['debt_to_income_ratio'] = df_processed['total_debt_outstanding'] / (df_processed['income'] + 1e-6)

    # Create Loan-to-Income Ratio
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
    # Define file paths
    input_data_path = 'data/Loan_Data.csv'
    output_data_path = 'data/processed'
    
    # 1. Load the raw data
    df = load_data(input_data_path)
    
    # 2. Perform initial, row-wise preprocessing (safe before splitting)
    df_featured = create_features(df.copy())
    
    # 3. Split data into features (X) and target (y)
    X = df_featured.drop('default', axis=1)
    y = df_featured['default']
    
    # 4. Split data into training and testing sets
    # We do this BEFORE any operations that learn from the data's distribution (like outlier bounds or scaling)
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print("Data splitting complete.")
    
    # 5. Handle outliers AFTER splitting
    # Learn the bounds from the training data, then apply them to both train and test sets.
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
    print("Outlier handling complete.")

    # 6. Save the final processed data
    save_processed_data(
        output_data_path,
        X_train=X_train,
        X_test=X_test,
        y_train=pd.DataFrame(y_train), # Saving series as DataFrame for consistency
        y_test=pd.DataFrame(y_test)
    )

if __name__ == '__main__':
    main()