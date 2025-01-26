import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from pyswarm import pso
import streamlit as st

# Load dataset
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_csv('https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv')
    return data

# Preprocess data
def preprocess_data(data):
    categorical_features = ['sex', 'smoker', 'region']
    numeric_features = ['age', 'bmi', 'children']

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])

    X = data.drop('charges', axis=1)
    y = data['charges']

    X_transformed = preprocessor.fit_transform(X)
    return X_transformed, y, preprocessor

# Define fitness function for PSO
def fitness_function(params, X_train, y_train, X_val, y_val):
    # Extract parameters
    intercept = params[0]
    coefficients = np.array(params[1:])

    # Predict using the linear regression model
    y_pred = np.dot(X_train, coefficients) + intercept

    # Calculate MSE on validation set
    mse = mean_squared_error(y_val, y_pred)
    return mse

# PSO Optimization
def optimize_pso(X_train, y_train, X_val, y_val):
    num_features = X_train.shape[1]

    # Lower and upper bounds for parameters
    lb = [-100] + [-10] * num_features
    ub = [100] + [10] * num_features

    # PSO optimization
    best_params, _ = pso(
        fitness_function, lb, ub, args=(X_train, y_train, X_val, y_val), swarmsize=30, maxiter=100
    )

    return best_params

# Main function for Streamlit
st.title('Insurance Cost Prediction with PSO')

# File upload option
uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type="csv")
data = load_data(uploaded_file)

st.write("### Dataset Preview")
st.dataframe(data.head())

# Preprocess data
X, y, preprocessor = preprocess_data(data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

# PSO Optimization
if st.button('Train Model with PSO'):
    st.write("### Training in Progress...")
    best_params = optimize_pso(X_train, y_train, X_val, y_val)

    # Extract optimal parameters
    intercept = best_params[0]
    coefficients = np.array(best_params[1:])

    # Make predictions
    y_pred = np.dot(X_test, coefficients) + intercept

    # Evaluate model
    mse = mean_squared_error(y_test, y_pred)
    st.write(f"### Model Evaluation")
    st.write(f"Mean Squared Error (MSE): {mse:.2f}")

    # Display predictions
    results = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred
    })
    st.write("### Predictions vs Actual")
    st.dataframe(results)
