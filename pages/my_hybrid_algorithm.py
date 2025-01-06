import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
# Ensure the path or package for 'my_hybrid_algorithm' is correct
try:
    from my_hybrid_algorithm import hybrid_ema_ga  # Replace with the correct import path
except ImportError as e:
    st.error(f"Failed to import hybrid_ema_ga: {e}")
    st.stop()

st.title("Hybrid Optimization Technique: EMA + GA")
st.sidebar.header("Algorithm Parameters")

# User inputs
population_size = st.sidebar.slider("Population Size", 10, 100, 50)
iterations = st.sidebar.slider("Iterations", 10, 500, 100)

st.write(f"Running EMA+GA with Population Size: {population_size}, Iterations: {iterations}")

# Execute the hybrid algorithm and handle errors gracefully
try:
    results = hybrid_ema_ga(population_size, iterations)
except Exception as e:
    st.error(f"Error running hybrid_ema_ga: {e}")
    st.stop()

# Display results
if results and 'fitness_progress' in results:
    st.line_chart(results['fitness_progress'])
else:
    st.error("No fitness progress data available.")

st.write("Performance Metrics")
if 'best_fitness' in results:
    st.write(f"Best Fitness: {results['best_fitness']}")
else:
    st.error("Best fitness data is missing.")

if 'execution_time' in results:
    st.write(f"Execution Time: {results['execution_time']} seconds")
else:
    st.error("Execution time data is missing.")
