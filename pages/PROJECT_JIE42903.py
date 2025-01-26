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
    df = pd.read_csv(uploaded_file)
    st.write("Preview of Uploaded Data")
    st.write(df.head())

    if st.button("Run PSO Optimization"):
        # Run PSO using the uploaded dataset
        try:
            results = run_pso_algorithm(df)
            
            # Display Results
            st.write("Optimization Results")
            st.write("Best Solution (Optimal Parameters):")
            st.json(results['best_solution'])  # Display as JSON for readability
            
            # Plot Fitness Trends
            st.subheader('Fitness Trends Over Generations')
            fig, ax = plt.subplots()
            ax.plot(results['generations'], results['fitness_values'])
            ax.set_xlabel('Generation')
            ax.set_ylabel('Fitness Value')
            ax.set_title('Fitness Trends')
            st.pyplot(fig)

            # Define a function to calculate predictions
            def calculate_prediction(row, solution):
                try:
                    # Handle intercept and coefficients dynamically
                    intercept = solution.get('intercept', 0)
                    age_coeff = solution.get('age', 0)
                    bmi_coeff = solution.get('bmi', 0)
                    smoker_coeff = solution.get('smoker', 0)

                    # Convert 'smoker' to numeric (1 for 'yes', 0 for 'no')
                    smoker_value = 1 if row['smoker'] == 'yes' else 0

                    # Prediction formula
                    return (
                        intercept +
                        age_coeff * row['age'] +
                        bmi_coeff * row['bmi'] +
                        smoker_coeff * smoker_value
                    )
                except KeyError as e:
                    raise ValueError(f"Missing key in solution: {e}")

            # Apply the best solution to predict costs
            try:
                df['Predicted_Cost'] = df.apply(lambda row: calculate_prediction(row, results['best_solution']), axis=1)
                st.subheader("Predicted Insurance Costs")
                if 'charges' in df.columns:
                    df['Error'] = abs(df['charges'] - df['Predicted_Cost'])
                    st.write(df[['charges', 'Predicted_Cost', 'Error']])
                else:
                    st.write(df[['Predicted_Cost']])
            except Exception as e:
                st.error(f"Error in calculating predictions: {e}")
        except Exception as e:
            st.error(f"Error during optimization: {e}")
