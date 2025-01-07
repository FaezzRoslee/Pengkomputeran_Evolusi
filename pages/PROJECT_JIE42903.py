import streamlit as st
from hybrid_ema_ga import hybrid_ema_ga, sample_fitness_function
import pandas as pd
import matplotlib.pyplot as plt

st.title("Hybrid EMA-GA Optimization Visualization")

st.sidebar.header("Parameters")
population_size = st.sidebar.slider("Population Size", 10, 100, 50)
generations = st.sidebar.slider("Generations", 10, 500, 100)

if st.button("Run Optimization"):
    final_population = hybrid_ema_ga(sample_fitness_function)
    st.write("Optimization Completed")
    # Visualization Example
    st.line_chart([ind.fitness.values[0] for ind in final_population])
