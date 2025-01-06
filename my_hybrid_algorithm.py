import numpy as np
import time

def hybrid_ema_ga(population_size, iterations):
    """Hybrid Exchange Market Algorithm and Genetic Algorithm."""
    np.random.seed(42)  # For reproducibility

    # Initialize population
    population = np.random.rand(population_size)
    fitness_progress = []

    start_time = time.time()

    # Example simplified hybrid algorithm
    for iteration in range(iterations):
        # EMA-inspired behavior: Exchange values between members
        for i in range(population_size):
            partner = np.random.randint(0, population_size)
            if population[i] > population[partner]:
                population[i] = (population[i] + population[partner]) / 2

        # Apply GA mutation
        mutation_rate = 0.01
        mutation_indices = np.random.choice(range(population_size), size=int(mutation_rate * population_size), replace=False)
        population[mutation_indices] += np.random.normal(0, 0.1, size=len(mutation_indices))

        # Track fitness (example: sum of squares minimization)
        fitness = np.sum(population**2)
        fitness_progress.append(fitness)

    end_time = time.time()
    execution_time = end_time - start_time

    best_fitness = np.min(fitness_progress)

    # Return results
    return {
        'fitness_progress': fitness_progress,
        'best_fitness': best_fitness,
        'execution_time': execution_time
    }
