import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from my_hybrid_algorithm import hybrid_ema_ga  # Custom implementation

st.title("Hybrid Optimization Technique: EMA + GA")
st.sidebar.header("Algorithm Parameters")

# Input parameters for the algorithm
population_size = st.sidebar.slider("Population Size", 10, 100, 50)
iterations = st.sidebar.slider("Iterations", 10, 500, 100)

st.write(f"Running EMA+GA with Population Size: {population_size}, Iterations: {iterations}")

# Execute the hybrid algorithm
try:
    results = hybrid_ema_ga(population_size, iterations)
    st.line_chart(results['fitness_progress'])
    st.write("Performance Metrics")
    st.write(f"Best Fitness: {results['best_fitness']}")
    st.write(f"Execution Time: {results['execution_time']} seconds")
except Exception as e:
    st.error(f"Error running hybrid_ema_ga: {e}")
