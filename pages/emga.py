import numpy as np
import random
import streamlit as st
import pandas as pd

# Define the fitness function (example: Sphere function)
def fitness_function(solution):
    return sum(x**2 for x in solution)

# Initialize population with random solutions
def initialize_population(pop_size, dimensions, bounds=(-100, 100)):
    return [np.random.uniform(bounds[0], bounds[1], dimensions) for _ in range(pop_size)]

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

# Main GA function
def genetic_algorithm(pop_size, dimensions, max_generations, bounds, mutation_rate):
    st.write(f"Running GA with population size: {pop_size}, dimensions: {dimensions}, generations: {max_generations}")
    population = initialize_population(pop_size, dimensions, bounds)
    best_solution = min(population, key=fitness_function)
    for generation in range(max_generations):

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
    st.title("Genetic Algorithm (GA)")

    # User inputs for parameters
    pop_size = st.number_input("Population Size", min_value=10, max_value=500, value=50, step=10)
    dimensions = st.number_input("Number of Dimensions", min_value=2, max_value=100, value=10, step=1)
    max_generations = st.number_input("Max Generations", min_value=1, max_value=1000, value=100, step=10)
    lower_bound = st.number_input("Lower Bound", value=-100.0)
    upper_bound = st.number_input("Upper Bound", value=100.0)
    mutation_rate = st.slider("Mutation Rate", min_value=0.0, max_value=1.0, value=0.05, step=0.01)

    # File uploader for dataset
    uploaded_file = st.file_uploader("Upload a CSV dataset", type="csv")
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.write("Uploaded Dataset:")
        st.dataframe(data)

        # Extract first row as a test solution
        if len(data.columns) > 0:
            test_solution = data.iloc[0].values
            test_fitness = fitness_function(test_solution)
            st.markdown(f"```
Fitness of first solution from dataset: {test_fitness:.4f}
```")

    # Add a button to run the algorithm
    if st.button("Run GA"):
        bounds = (lower_bound, upper_bound)
        best = genetic_algorithm(pop_size, dimensions, max_generations, bounds, mutation_rate)

        # Format the output as a box
        formatted_solution = "Best solution found:\n" + " | ".join([f"x{i} = {value:.4f}" for i, value in enumerate(best)])
        st.markdown(f"```
{formatted_solution}
```")

        # Display the fitness of the best solution
        best_fitness = fitness_function(best)
        st.markdown(f"```
Fitness of best solution: {best_fitness:.4f}
```")
