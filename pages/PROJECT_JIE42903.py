import streamlit as st
from hybrid_optimizer import hybrid_ema_ga, fitness_function

st.title("Hybrid Optimization: Exchange Market Algorithm and Genetic Algorithm")

st.sidebar.header("Settings")
pop_size = st.sidebar.slider("Population Size", 10, 100, 50)
dimensions = st.sidebar.slider("Dimensions", 2, 10, 5)
iterations = st.sidebar.slider("Iterations", 10, 200, 100)

if st.button("Run Optimization"):
    best_solution = hybrid_ema_ga(pop_size, dimensions, iterations)
    st.write(f"Best Solution: {best_solution}")
    st.write(f"Best Fitness: {fitness_function(best_solution)}")
