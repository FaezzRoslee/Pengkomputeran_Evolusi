import streamlit as st
import pandas as pd
import random
import time

# Streamlit setup
st.set_page_config(page_title="Genetic Algorithm for Scheduling")
st.header("Genetic Algorithm for Program Scheduling")

# File Upload
@st.cache
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

uploaded_file = st.file_uploader("Upload Program Ratings CSV", type=["csv"])
if uploaded_file:
    data = load_data(uploaded_file)
    st.subheader("Program Ratings")
    st.dataframe(data)

    # Genetic Algorithm Parameters
    with st.sidebar:
        st.subheader("Genetic Algorithm Parameters")
        POP_SIZE = st.number_input("Population Size", min_value=100, max_value=1000, value=500, step=50)
        MUT_RATE = st.slider("Mutation Rate", min_value=0.01, max_value=0.05, value=0.02, step=0.01)
        TARGET = st.text_input("Target String (e.g., optimal program schedule)", "news sports documentary")

    GENES = list(data.columns[1:])  # Use columns (e.g., hours) as genes

    # Helper Functions
    def initialize_pop(target_len):
        population = []
        for _ in range(POP_SIZE):
            chromosome = [random.choice(GENES) for _ in range(target_len)]
            population.append(chromosome)
        return population

    def fitness_cal(target, chromosome):
        return sum(1 for t, c in zip(target.split(), chromosome) if t != c)

    def selection(population, fitness):
        fitness_sorted = sorted(zip(population, fitness), key=lambda x: x[1])
        return [ch[0] for ch in fitness_sorted[:POP_SIZE // 2]]

    def crossover(parent1, parent2):
        point = random.randint(1, len(parent1) - 1)
        return parent1[:point] + parent2[point:]

    def mutate(chromosome):
        for i in range(len(chromosome)):
            if random.random() < MUT_RATE:
                chromosome[i] = random.choice(GENES)
        return chromosome

    # Main Function
    def genetic_algorithm(target):
        target_len = len(target.split())
        population = initialize_pop(target_len)
        generation = 0

        while True:
            fitness = [fitness_cal(target, chromosome) for chromosome in population]

            # Check if target is achieved
            if min(fitness) == 0:
                best_solution = population[fitness.index(min(fitness))]
                return best_solution, generation

            # Selection
            selected = selection(population, fitness)

            # Crossover and Mutation
            offspring = []
            for _ in range(POP_SIZE):
                parent1, parent2 = random.sample(selected, 2)
                child = crossover(parent1, parent2)
                child = mutate(child)
                offspring.append(child)

            population = offspring
            generation += 1

    # Run the Genetic Algorithm
    if st.button("Run Algorithm"):
        with st.spinner("Running Genetic Algorithm..."):
            start_time = time.time()
            solution, generations = genetic_algorithm(TARGET)
            elapsed_time = time.time() - start_time

            st.success("Optimization Complete!")
            st.write(f"Solution: {solution}")
            st.write(f"Generations: {generations}")
            st.write(f"Elapsed Time: {elapsed_time:.2f} seconds")

else:
    st.warning("Please upload the CSV file to proceed.")
