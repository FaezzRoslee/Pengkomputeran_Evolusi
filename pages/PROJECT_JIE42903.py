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
    return [ind + np.random.uniform(-mode_factor, mode_factor, size=ind.shape) for ind, _ in group]

# Main EMGA function
def emga(population_size, num_generations, crossover_rate, mutation_rate, termination_criteria):
    population = initialize_population(population_size)
    for generation in range(num_generations):
        group_1, group_2, group_3 = rank_and_group(population)
        
        # Balanced mode on second and third group
        group_2 = apply_mode_changes(group_2, mode_factor=1)
        group_3 = apply_mode_changes(group_3, mode_factor=2)
        
        # Oscillation mode
        group_2 = apply_mode_changes(group_2, mode_factor=4+8)
        group_3 = apply_mode_changes(group_3, mode_factor=9+11)
        
        # Apply crossover and mutation
        new_population = crossover_and_mutate(group_1 + group_2 + group_3, crossover_rate, mutation_rate)
        
        # Combine populations and select the best
        population = select_best_population(population + new_population, population_size)
        
        # Termination check
        if min(cost_function(ind) for ind in population) <= termination_criteria:
            st.write(f"Termination reached at generation {generation}")
            break
    
    best_solution = min(population, key=cost_function)
    st.write("Best Solution:", best_solution)
    st.write("Best Cost:", cost_function(best_solution))

# Placeholder for crossover and mutation (simplified)
def crossover_and_mutate(population, crossover_rate, mutation_rate):
    new_population = []
    for _ in range(len(population) // 2):
        p1, p2 = random.sample(population, 2)
        if random.random() < crossover_rate:
            point = random.randint(1, len(p1) - 1)
            child1 = np.concatenate((p1[:point], p2[point:]))
            child2 = np.concatenate((p2[:point], p1[point:]))
        else:
            child1, child2 = p1, p2
        new_population.extend([mutate(child1, mutation_rate), mutate(child2, mutation_rate)])
    return new_population

def mutate(individual, mutation_rate):
    if random.random() < mutation_rate:
        idx = random.randint(0, len(individual) - 1)
        individual[idx] += np.random.normal()
    return individual

def select_best_population(population, size):
    return sorted(population, key=cost_function)[:size]

# Run the algorithm
if st.button("Run EMGA"):
    emga(population_size, num_generations, crossover_rate, mutation_rate, termination_criteria)
