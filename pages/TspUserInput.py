import matplotlib.pyplot as plt
from itertools import permutations, combinations
import random
import numpy as np
import seaborn as sns
import streamlit as st

# Streamlit inputs for cities and coordinates
city_names = []
x_coords = []
y_coords = []

st.title("Traveling Salesman Problem with Genetic Algorithm")
st.subheader("Input the Cities with Coordinates")

for i in range(10):
    city_name = st.text_input(f"Enter city name #{i + 1}:", key=f"city_name_{i}")
    x_coord = st.number_input(f"Enter x-coordinate for {city_name}:", key=f"x_coord_{i}")
    y_coord = st.number_input(f"Enter y-coordinate for {city_name}:", key=f"y_coord_{i}")
    
    city_names.append(city_name)
    x_coords.append(x_coord)
    y_coords.append(y_coord)

# Zip the inputs into coordinates
city_coords = dict(zip(city_names, zip(x_coords, y_coords)))

# Constants for the Genetic Algorithm
n_population = 250
crossover_per = 0.8
mutation_per = 0.2
n_generations = 200

# Generate pastel colors for each city
colors = sns.color_palette("pastel", len(city_names))
# Sample icons for cities
city_icons = {city: f"♛" for city in city_names}

# Plot initial city graph
fig, ax = plt.subplots()
ax.grid(False)

for i, (city, (city_x, city_y)) in enumerate(city_coords.items()):
    color = colors[i]
    icon = city_icons[city]
    ax.scatter(city_x, city_y, c=[color], s=1200, zorder=2)
    ax.annotate(icon, (city_x, city_y), fontsize=40, ha='center', va='center', zorder=3)
    ax.annotate(city, (city_x, city_y), fontsize=12, ha='center', va='bottom', xytext=(0, -30), textcoords='offset points')

    # Connect cities with opaque lines
    for j, (other_city, (other_x, other_y)) in enumerate(city_coords.items()):
        if i != j:
            ax.plot([city_x, other_x], [city_y, other_y], color='gray', linestyle='-', linewidth=1, alpha=0.1)

fig.set_size_inches(16, 12)
st.pyplot(fig)

# Functions for Genetic Algorithm
def initial_population(cities_list, n_population=250):
    population_perms = []
    possible_perms = list(permutations(cities_list))
    random_ids = random.sample(range(len(possible_perms)), n_population)
    for i in random_ids:
        population_perms.append(list(possible_perms[i]))
    return population_perms

def dist_two_cities(city_1, city_2):
    city_1_coords = city_coords[city_1]
    city_2_coords = city_coords[city_2]
    return np.sqrt(np.sum((np.array(city_1_coords) - np.array(city_2_coords))**2))

def total_dist_individual(individual):
    total_dist = 0
    for i in range(len(individual)):
        if i == len(individual) - 1:
            total_dist += dist_two_cities(individual[i], individual[0])
        else:
            total_dist += dist_two_cities(individual[i], individual[i + 1])
    return total_dist

def fitness_prob(population):
    total_dist_all_individuals = [total_dist_individual(ind) for ind in population]
    max_population_cost = max(total_dist_all_individuals)
    population_fitness = max_population_cost - np.array(total_dist_all_individuals)
    population_fitness_sum = sum(population_fitness)
    population_fitness_probs = population_fitness / population_fitness_sum
    return population_fitness_probs

def roulette_wheel(population, fitness_probs):
    population_fitness_probs_cumsum = fitness_probs.cumsum()
    selected_individual_index = np.searchsorted(population_fitness_probs_cumsum, np.random.rand())
    return population[selected_individual_index]

def crossover(parent_1, parent_2):
    n_cities_cut = len(city_names) - 1
    cut = round(random.uniform(1, n_cities_cut))
    offspring_1 = parent_1[:cut] + [city for city in parent_2 if city not in parent_1[:cut]]
    offspring_2 = parent_2[:cut] + [city for city in parent_1 if city not in parent_2[:cut]]
    return offspring_1, offspring_2

def mutation(offspring):
    """
    Implement mutation strategy in a single offspring
    Input:
    1- offspring individual
    Output:
    1- mutated offspring individual
    """
    n_cities_cut = len(cities_names) - 1
    index_1 = random.randint(0, n_cities_cut)
    index_2 = random.randint(0, n_cities_cut)

    # Swap the cities at index_1 and index_2
    offspring[index_1], offspring[index_2] = offspring[index_2], offspring[index_1]
    return offspring

def run_ga(cities_list, n_population, n_generations, crossover_per, mutation_per):
    population = initial_population(cities_list, n_population)
    for _ in range(n_generations):
        fitness_probs = fitness_prob(population)
        parents_list = [roulette_wheel(population, fitness_probs) for _ in range(int(crossover_per * n_population))]
        
        offspring_list = []
        for i in range(0, len(parents_list), 2):
            offspring_1, offspring_2 = crossover(parents_list[i], parents_list[i + 1])
            if random.random() < mutation_per:
                offspring_1 = mutation(offspring_1)
            if random.random() < mutation_per:
                offspring_2 = mutation(offspring_2)
            offspring_list.extend([offspring_1, offspring_2])

        population = parents_list + offspring_list
        population = sorted(population, key=total_dist_individual)[:n_population]

    return population

# Running the Genetic Algorithm
best_mixed_offspring = run_ga(city_names, n_population, n_generations, crossover_per, mutation_per)
total_dist_all_individuals = [total_dist_individual(ind) for ind in best_mixed_offspring]
index_minimum = np.argmin(total_dist_all_individuals)
minimum_distance = min(total_dist_all_individuals)

# Display results
st.write("Minimum Distance:", minimum_distance)
st.write("Shortest Path:", best_mixed_offspring[index_minimum])

# Plot the shortest path
x_shortest = [city_coords[city][0] for city in best_mixed_offspring[index_minimum]]
y_shortest = [city_coords[city][1] for city in best_mixed_offspring[index_minimum]]
x_shortest.append(x_shortest[0])
y_shortest.append(y_shortest[0])

fig, ax = plt.subplots()
ax.plot(x_shortest, y_shortest, '--go', label='Best Route', linewidth=2.5)
plt.legend()

plt.title("TSP Best Route Using GA", fontsize=25)
plt.suptitle(f"Total Distance Travelled: {round(minimum_distance, 3)}\n"
             f"{n_generations} Generations, {n_population} Population Size\n"
             f"Crossover: {crossover_per}, Mutation: {mutation_per}",
             fontsize=18, y=1.05)

for i, city in enumerate(best_mixed_offspring[index_minimum]):
    ax.annotate(f"{i + 1}- {city}", (x_shortest[i], y_shortest[i]), fontsize=20)

fig.set_size_inches(16, 12)
st.pyplot(fig)
