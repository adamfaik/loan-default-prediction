# app.py
from flask import Flask, request, render_template
import pandas as pd
import joblib
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model and the scaler
model = joblib.load('models/model.pkl')
scaler = joblib.load('processors/scaler.joblib')

# Define the home page route
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the form
    input_features = [float(x) for x in request.form.values()]

    # The order of features must match the order the model was trained on.
    # This order is from our X_train.csv file.
    feature_names = ['credit_lines_outstanding', 'loan_amt_outstanding', 
                     'total_debt_outstanding', 'income', 'years_employed', 'fico_score',
                     'debt_to_income_ratio', 'loan_to_income_ratio']

    # We need to re-create the engineered features from the raw inputs
    # Note: The form gives us inputs in a different order. We need to map them correctly.
    form_values = request.form

    # Extract basic features in the correct order
    # This is a bit manual and depends on your HTML form's 'name' attributes
    # A more robust app might handle this mapping more elegantly.
    credit_lines = float(form_values['credit_lines_outstanding'])
    loan_amt = float(form_values['loan_amt_outstanding'])
    total_debt = float(form_values['total_debt_outstanding'])
    income = float(form_values['income'])
    years_employed = float(form_values['years_employed'])
    fico_score = float(form_values['fico_score'])

    # Re-engineer the ratio features
    debt_to_income = total_debt / (income + 1e-6)
    loan_to_income = loan_amt / (income + 1e-6)

    # Create the final feature array in the correct order for the model
    final_features = np.array([credit_lines, loan_amt, total_debt, income, 
                               years_employed, fico_score, debt_to_income, loan_to_income])

    # Reshape for a single prediction and scale the data
    final_features_reshaped = final_features.reshape(1, -1)
    scaled_features = scaler.transform(final_features_reshaped)

    # Make the prediction
    prediction = model.predict(scaled_features)

    # Display the result
    output = "Will Default" if prediction[0] == 1 else "Will Not Default"

    return render_template('index.html', prediction_text=f'Predicted Status: {output}')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)