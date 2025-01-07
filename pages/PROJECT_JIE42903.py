import streamlit as st
import numpy as np
import random

# Title
st.title("Hybrid Optimization Algorithm (EMGA)")

# Inputs
population_size = st.number_input("Enter population size:", min_value=10, step=1)
num_generations = st.number_input("Enter number of generations:", min_value=10, step=1)
crossover_rate = st.slider("Crossover rate:", 0.0, 1.0, 0.8)
mutation_rate = st.slider("Mutation rate:", 0.0, 1.0, 0.1)
termination_criteria = st.number_input("Termination threshold (or max fitness):", min_value=0.1)

# Initialize population
def initialize_population(size):
    return [np.random.uniform(low=-10, high=10, size=5) for _ in range(size)]

# Cost function (example)
def cost_function(individual):
    return np.sum(individual**2)

# Ranking and grouping
def rank_and_group(population):
    costs = [(ind, cost_function(ind)) for ind in population]
    sorted_population = sorted(costs, key=lambda x: x[1])
    group_1 = sorted_population[:len(population)//3]
    group_2 = sorted_population[len(population)//3:2*len(population)//3]
    group_3 = sorted_population[2*len(population)//3:]
    return group_1, group_2, group_3

# Balanced and Oscillation Modes
def apply_mode_changes(group, mode_factor):
    # Check if group is a list of tuples and extract the individual solution (assuming it's a tuple: (individual, cost))
    return [
        ind + np.random.uniform(-mode_factor, mode_factor, size=ind.shape)  # Assuming 'ind' is a numpy array
        for ind, _ in group  # _ represents the cost in the tuple (ind, cost)
    ]

# Modify the group population before using it
def rank_and_group(population):
    costs = [(ind, cost_function(ind)) for ind in population]
    sorted_population = sorted(costs, key=lambda x: x[1])  # Sort by cost
    group_1 = sorted_population[:len(population)//3]
    group_2 = sorted_population[len(population)//3:2*len(population)//3]
    group_3 = sorted_population[2*len(population)//3:]
    return group_1, group_2, group_3

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
