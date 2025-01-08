import numpy as np
import random
import streamlit as st

# Define the fitness function (example: Sphere function)
def fitness_function(solution):
    return sum(x**2 for x in solution)

# Initialize population with random solutions
def initialize_population(pop_size, dimensions, bounds=(-100, 100)):
    return [np.random.uniform(bounds[0], bounds[1], dimensions) for _ in range(pop_size)]

# Categorize stockholders based on balanced and fluctuating market states
def categorize_population(population, fitness_fn, state="balanced"):
    population.sort(key=fitness_fn)
    if state == "balanced":
        return (
            population[:len(population) // 4],  # 25% first group
            population[len(population) // 4 : len(population) // 4 + len(population) // 2],  # 50% second group
            population[len(population) // 4 + len(population) // 2 :]  # 25% third group
        )
    elif state == "fluctuating":
        return (
            population[:len(population) // 5],  # 20% first group
            population[len(population) // 5 : len(population) // 5 + len(population) * 3 // 5],  # 60% second group
            population[len(population) // 5 + len(population) * 3 // 5 :]  # 20% third group
        )

# Exchange market algorithm operation
def exchange_market_operation(population_groups, risk1, risk2, mode="balanced"):
    second_group, third_group = population_groups[1], population_groups[2]
    new_second_group = [individual + risk1 * np.random.uniform(-1, 1, len(individual)) for individual in second_group]
    new_third_group = [individual + risk2 * np.random.uniform(-1, 1, len(individual)) for individual in third_group]
    return population_groups[0], new_second_group, new_third_group

# Genetic Algorithm crossover and mutation
def crossover_and_mutation(population, mutation_rate=0.05):
    next_generation = []
    for _ in range(len(population) // 2):
        parent1, parent2 = random.sample(population, 2)
        cross_point = random.randint(1, len(parent1) - 1)
        child1 = np.concatenate((parent1[:cross_point], parent2[cross_point:]))
        child2 = np.concatenate((parent2[:cross_point], parent1[cross_point:]))
        next_generation.extend([mutate(child1, mutation_rate), mutate(child2, mutation_rate)])
    return next_generation

def mutate(solution, rate):
    """Apply mutation with a given rate."""
    return [gene + np.random.normal() if random.random() < rate else gene for gene in solution]

# Main EMGA function
def emga(pop_size, dimensions, max_generations, state, bounds, risk1, risk2, mutation_rate):
    st.write(f"Running EMGA with population size: {pop_size}, dimensions: {dimensions}, generations: {max_generations}")
    population = initialize_population(pop_size, dimensions, bounds)
    best_solution = min(population, key=fitness_function)
    for generation in range(max_generations):
        # Categorize population based on market state
        population_groups = categorize_population(population, fitness_function, state)
        
        # Apply Exchange Market operations
        first_group, second_group, third_group = exchange_market_operation(population_groups, risk1, risk2, state)
        
        # Combine groups and rank
        population = first_group + second_group + third_group
        population.sort(key=fitness_function)
        
        # Apply Genetic Algorithm operations
        ga_population = crossover_and_mutation(population, mutation_rate)
        ga_population.sort(key=fitness_function)
        
        # Combine old and new populations, select best half
        combined_population = population + ga_population
        combined_population.sort(key=fitness_function)
        population = combined_population[:pop_size]
        
        # Track the best solution
        current_best = min(population, key=fitness_function)
        if fitness_function(current_best) < fitness_function(best_solution):
            best_solution = current_best

        st.write(f"Generation {generation+1}: Best fitness = {fitness_function(best_solution)}")

    return best_solution

if __name__ == "__main__":
    st.title("Hybrid Exchange Market Genetic Algorithm (EMGA)")
    
    # User inputs for parameters
    pop_size = st.number_input("Population Size", min_value=10, max_value=500, value=50, step=10)
    dimensions = st.number_input("Number of Dimensions", min_value=2, max_value=100, value=10, step=1)
    max_generations = st.number_input("Max Generations", min_value=1, max_value=1000, value=100, step=10)
    state = st.selectbox("Market State", ["balanced", "fluctuating"])
    lower_bound = st.number_input("Lower Bound", value=-100.0)
    upper_bound = st.number_input("Upper Bound", value=100.0)
    risk1 = st.slider("Risk Factor 1", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
    risk2 = st.slider("Risk Factor 2", min_value=0.0, max_value=1.0, value=0.2, step=0.01)
    mutation_rate = st.slider("Mutation Rate", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
    
    # Add a button to run the algorithm
    if st.button("Run EMGA"):
        bounds = (lower_bound, upper_bound)
        best = emga(pop_size, dimensions, max_generations, state, bounds, risk1, risk2, mutation_rate)
        st.write(f"Best solution found: {best}")
