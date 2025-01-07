import streamlit as st
import numpy as np

# Function to initialize population
def initialize_population(population_size):
    return [np.random.rand(10) for _ in range(population_size)]  # Example of initializing 10-dimensional solutions

# Function to calculate the cost of an individual
def cost_function(individual):
    return np.sum(individual**2)  # Simple cost function, you can replace it with your own

# Function to apply mode changes (correctly handling numpy arrays)
def apply_mode_changes(group, mode_factor):
    return [
        ind + np.random.uniform(-mode_factor, mode_factor, size=ind.shape)  # Assuming 'ind' is a numpy array
        for ind, _ in group  # _ represents the cost in the tuple (ind, cost)
    ]

# Function for ranking and grouping population
def rank_and_group(population):
    costs = [(ind, cost_function(ind)) for ind in population]
    sorted_population = sorted(costs, key=lambda x: x[1])  # Sort by cost
    group_1 = sorted_population[:len(population)//3]
    group_2 = sorted_population[len(population)//3:2*len(population)//3]
    group_3 = sorted_population[2*len(population)//3:]
    return group_1, group_2, group_3

# Function for crossover and mutation (for simplicity, just returning population)
def crossover_and_mutate(population, crossover_rate, mutation_rate):
    return population  # Placeholder, you can replace with actual crossover and mutation logic

# Function to select the best population (for now just returning the top 'population_size' individuals)
def select_best_population(population, population_size):
    return population[:population_size]

# EMGA function
def emga(population_size, num_generations, crossover_rate, mutation_rate, termination_criteria):
    population = initialize_population(population_size)
    for generation in range(num_generations):
        group_1, group_2, group_3 = rank_and_group(population)
        
        # Balanced mode on second and third group
        group_2 = apply_mode_changes(group_2, mode_factor=1)
        group_3 = apply_mode_changes(group_3, mode_factor=2)
        
        # Oscillation mode
        group_2 = apply_mode_changes(group_2, mode_factor=12)  # Combined oscillation effect
        group_3 = apply_mode_changes(group_3, mode_factor=20)  # Combined oscillation effect
        
        # Apply crossover and mutation
        new_population = crossover_and_mutate(group_1 + group_2 + group_3, crossover_rate, mutation_rate)
        
        # Combine populations and select the best
        population = select_best_population(new_population, population_size)
        
        # Termination check
        if min(cost_function(ind[0]) for ind in population) <= termination_criteria:
            st.write(f"Termination reached at generation {generation}")
            break
    
    best_solution = min(population, key=lambda x: cost_function(x[0]))
    st.write("Best Solution:", best_solution[0])
    st.write("Best Cost:", best_solution[1])

# Streamlit UI elements
st.title("Evolutionary Multi-Group Algorithm (EMGA)")

# Set the parameters for the algorithm
population_size = st.number_input("Population Size", min_value=10, max_value=100, value=50)
num_generations = st.number_input("Number of Generations", min_value=1, max_value=100, value=50)
crossover_rate = st.slider("Crossover Rate", 0.0, 1.0, 0.7)
mutation_rate = st.slider("Mutation Rate", 0.0, 1.0, 0.1)
termination_criteria = st.number_input("Termination Criteria", min_value=0.0, value=0.01)

# Button to run the EMGA algorithm
if st.button("Run EMGA"):
    emga(population_size, num_generations, crossover_rate, mutation_rate, termination_criteria)
