import streamlit as st
import pandas as pd
import random
import time

# Streamlit setup
st.set_page_config(page_title="Genetic Algorithm for Scheduling")
st.header("Genetic Algorithm for Program Scheduling")

# File Upload
@st.cache_data
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

uploaded_file = st.file_uploader("Upload Program Ratings CSV", type=["csv"])
if uploaded_file:
    data = load_data(uploaded_file)
    st.subheader("Program Ratings")
    st.dataframe(data)

    if "Type of Program" not in data.columns:
        st.error("The uploaded file must contain a 'Type of Program' column.")
        st.stop()

    PROGRAMS = data["Type of Program"].tolist()  # List of program names
    HOURS = data.columns[1:].tolist()  # List of hours (e.g., 'Hour 6', 'Hour 7', etc.)
    preferences = data.iloc[:, 1:].values  # Preference scores as a 2D array

    # Genetic Algorithm Parameters
    with st.sidebar:
        st.subheader("Genetic Algorithm Parameters")
        CO_R = st.slider("Crossover Rate (CO_R)", min_value=0.0, max_value=0.95, value=0.8, step=0.05)
        MUT_R = st.slider("Mutation Rate (MUT_R)", min_value=0.01, max_value=0.05, value=0.02, step=0.01)

    POP_SIZE = 500  # Fixed population size

    # Helper Functions
    def initialize_pop():
        population = []
        for _ in range(POP_SIZE):
            chromosome = random.sample(HOURS, len(PROGRAMS))  # Assign random hours to programs
            population.append(chromosome)
        return population

    def fitness_cal(chromosome):
        fitness = 0
        for i, hour in enumerate(chromosome):
            hour_index = HOURS.index(hour)
            fitness += preferences[i][hour_index]  # Add the preference score for the assigned hour
        return fitness

    def selection(population, fitness):
        fitness_sorted = sorted(zip(population, fitness), key=lambda x: x[1], reverse=True)
        return [ch[0] for ch in fitness_sorted[:POP_SIZE // 2]]

    def crossover(parent1, parent2):
        if random.random() < CO_R:
            point = random.randint(1, len(parent1) - 1)
            return parent1[:point] + parent2[point:]
        return parent1

    def mutate(chromosome):
        for i in range(len(chromosome)):
            if random.random() < MUT_R:
                chromosome[i] = random.choice(HOURS)
        return chromosome

    # Main Function
    def genetic_algorithm():
        population = initialize_pop()
        generation = 0

        while generation < 100:  # Stop after a fixed number of generations for simplicity
            fitness = [fitness_cal(chromosome) for chromosome in population]

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

        # Return the best solution
        best_fitness = max(fitness)
        best_solution = population[fitness.index(best_fitness)]
        return best_solution

    # Run the Genetic Algorithm
    if st.button("Run Algorithm"):
        with st.spinner("Running Genetic Algorithm..."):
            solution = genetic_algorithm()

            # Display results
            st.success("Optimization Complete!")

            # Display Parameters
            st.subheader("Parameters Used")
            st.write(f"- **Crossover Rate (CO_R):** {CO_R}")
            st.write(f"- **Mutation Rate (MUT_R):** {MUT_R}")

            # Display Schedule
            st.subheader("Resulting Schedule")
            schedule_df = pd.DataFrame({
                "Program": PROGRAMS,
                "Assigned Hour": solution
            })
            st.dataframe(schedule_df)

else:
    st.warning("Please upload the CSV file to proceed.")
