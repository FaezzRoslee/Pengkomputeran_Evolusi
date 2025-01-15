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
        st.write(results['best_solution'])  # Replace with relevant result structure
        
        st.subheader('Fitness Trends Over Generations')
        fig, ax = plt.subplots()
        ax.plot(results['generations'], results['fitness_values'])
        ax.set_xlabel('Generation')
        ax.set_ylabel('Fitness Value')
        st.pyplot(fig)
