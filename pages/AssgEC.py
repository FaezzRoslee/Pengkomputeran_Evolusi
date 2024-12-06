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

    # Extract program names
    if "Type of Program" in data.columns:
        GENES = list(data["Type of Program"])  # Use "Type of Program" column as GENES
        TARGET = GENES  # Target schedule is the same as the program names
    else:
        st.error("The uploaded file must contain a 'Type of Program' column.")
        st.stop()

    # Genetic Algorithm Parameters
    with st.sidebar:
        st.subheader("Genetic Algorithm Parameters")
        CO_R = st.slider("Crossover Rate (CO_R)", min_value=0.0, max_value=0.95, value=0.8, step=0.05)
        MUT_R = st.slider("Mutation Rate (MUT_R)", min_value=0.01, max_value=0.05, value=0.02, step=0.01)
    
    POP_SIZE = 500  # Fixed population size

    # Helper Functions
    def initialize_pop(target_len):
        population = []
        for _ in range(POP_SIZE):
            chromosome = [random.choice(GENES) for _ in range(target_len)]
            population.append(chromosome)
        return population

    def fitness_cal(target, chromosome):
        return sum(1 for t, c in zip(target, chromosome) if t != c)

    def selection(population, fitness):
        fitness_sorted = sorted(zip(population, fitness), key=lambda x: x[1])
        return [ch[0] for ch in fitness_sorted[:POP_SIZE // 2]]

    def crossover(parent1, parent2):
        if random.random() < CO_R:
            point = random.randint(1, len(parent1) - 1)
            return parent1[:point] + parent2[point:]
        return parent1

    def mutate(chromosome):
        for i in range(len(chromosome)):
            if random.random() < MUT_R:
                chromosome[i] = random.choice(GENES)
        return chromosome

    # Main Function
    def genetic_algorithm(target):
        target_len = len(target)
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

            # Display results
            st.success("Optimization Complete!")
            st.write(f"Generations: {generations}")
            st.write(f"Elapsed Time: {elapsed_time:.2f} seconds")
            
            # Display Parameters
            st.subheader("Parameters Used")
            st.write(f"- **Crossover Rate (CO_R):** {CO_R}")
            st.write(f"- **Mutation Rate (MUT_R):** {MUT_R}")
            
            # Display Schedule
            st.subheader("Resulting Schedule")
            schedule_df = pd.DataFrame({
                "Program": GENES,  # Original "Type of Program" values
                "Scheduled Slot": solution
            })
            st.dataframe(schedule_df)

else:
    st.warning("Please upload the CSV file to proceed.")
