import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pso_algorithm import run_pso_algorithm  # Import from your GitHub-pushed module

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
        results = run_pso_algorithm(df)
        
        # Display Results
        st.write("Optimization Results")
        st.write("Best Solution (Optimal Parameters):", results['best_solution'])
        
        # Plot Fitness Trends
        st.subheader('Fitness Trends Over Generations')
        fig, ax = plt.subplots()
        ax.plot(results['generations'], results['fitness_values'])
        ax.set_xlabel('Generation')
        ax.set_ylabel('Fitness Value')
        st.pyplot(fig)

        # Define a function to calculate predictions
        def calculate_prediction(row, solution):
            return (
                solution[0] +  # Intercept
                solution[1] * row['Age'] +
                solution[2] * row['BMI'] +
                solution[3] * row['Smoker']
            )
        
        # Apply the best solution to predict costs
        try:
            df['Predicted_Cost'] = df.apply(lambda row: calculate_prediction(row, results['best_solution']), axis=1)
            st.subheader("Predicted Insurance Costs")
            if 'Actual_Cost' in df.columns:
                df['Error'] = abs(df['Actual_Cost'] - df['Predicted_Cost'])
                st.write(df[['Actual_Cost', 'Predicted_Cost', 'Error']])
            else:
                st.write(df[['Predicted_Cost']])
        except Exception as e:
            st.error(f"Error in calculating predictions: {e}")
