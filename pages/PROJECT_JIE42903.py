import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pso_algorithm import run_pso_algorithm  # Import from your module

# Title and Description
st.title('PSO Optimization for Insurance Cost Estimation')
st.write("This application demonstrates Particle Swarm Optimization (PSO) applied to insurance cost prediction.")

# Data Upload Section
uploaded_file = st.file_uploader("Upload your insurance dataset", type=["csv"])
if uploaded_file is not None:
    try:
        # Read and display the dataset
        df = pd.read_csv(uploaded_file)
        st.write("Preview of Uploaded Data")
        st.write(df.head())

        # Preprocessing step: Handle potential data issues
        st.write("Preprocessing dataset...")
        
        # Convert categorical 'smoker' column to numeric
        if 'smoker' in df.columns:
            df['smoker'] = df['smoker'].apply(lambda x: 1 if str(x).lower() == 'yes' else 0 if str(x).lower() == 'no' else None)
        else:
            st.warning("Column 'smoker' not found in the dataset.")

        # Convert necessary columns to numeric
        for column in ['age', 'bmi', 'charges']:
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors='coerce')  # Convert to numeric, invalid values become NaN

        # Drop rows with missing or invalid values
        df = df.dropna()

        # Ensure all columns used in PSO are numeric
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        st.write("Numeric Columns for Optimization:", numeric_columns)

        if df.empty:
            st.error("No valid data available after preprocessing. Please check your dataset.")
        else:
            st.write("Preprocessed Data:")
            st.write(df.head())

            # Button to run PSO Optimization
            if st.button("Run PSO Optimization"):
                try:
                    # Run PSO algorithm
                    results = run_pso_algorithm(df)

                    # Display optimization results
                    st.write("Optimization Results")
                    st.json(results['best_solution'])

                    # Plot Fitness Trends
                    st.subheader('Fitness Trends Over Generations')
                    fig, ax = plt.subplots()
                    ax.plot(results['generations'], results['fitness_values'])
                    ax.set_xlabel('Generation')
                    ax.set_ylabel('Fitness Value')
                    ax.set_title('Fitness Trends')
                    st.pyplot(fig)

                    # Predictions using best solution
                    def calculate_prediction(row, solution):
                        intercept = solution.get('intercept', 0)
                        age_coeff = solution.get('age', 0)
                        bmi_coeff = solution.get('bmi', 0)
                        smoker_coeff = solution.get('smoker', 0)
                        return (
                            intercept +
                            age_coeff * row['age'] +
                            bmi_coeff * row['bmi'] +
                            smoker_coeff * row['smoker']
                        )

                    df['Predicted_Cost'] = df.apply(lambda row: calculate_prediction(row, results['best_solution']), axis=1)

                    if 'charges' in df.columns:
                        df['Error'] = abs(df['charges'] - df['Predicted_Cost'])
                        st.write(df[['charges', 'Predicted_Cost', 'Error']])
                    else:
                        st.write(df[['Predicted_Cost']])
                except Exception as e:
                    st.error(f"Error during optimization: {e}")
    except Exception as e:
        st.error(f"Error loading or processing the file: {e}")
