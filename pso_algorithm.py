import numpy as np
import random
import pandas as pd

def fitness_cal(individual, data):
    """Calculates fitness as the negative absolute error between predicted and actual charges."""
    age, bmi, children, smoker = individual['age'], individual['bmi'], individual['children'], individual['smoker']
    # Sample cost estimation formula for demonstration
    cost = (250 * age) + (350 * bmi) + (5000 * children) + (20000 if smoker == 'yes' else 0)
    actual = data['charges'].mean()  # Replace with appropriate comparison if needed
    return -abs(cost - actual)

def initialize_particles(data, population_size=30):
    """Initializes a population of particles with random attributes."""
    particles = []
    for _ in range(population_size):
        particle = {
            'age': random.randint(18, 64),
            'bmi': random.uniform(15.0, 40.0),
            'children': random.randint(0, 5),
            'smoker': random.choice(['yes', 'no']),
            'fitness': float('-inf')
        }
        particles.append(particle)
    return particles

def update_velocity(particle, personal_best, global_best, w=0.5, c1=1.5, c2=1.5):
    """Updates particle velocity and returns new attributes."""
    velocity = {}
    for attr in ['age', 'bmi', 'children']:
        inertia = w * (random.random())
        cognitive = c1 * random.random() * (personal_best[attr] - particle[attr])
        social = c2 * random.random() * (global_best[attr] - particle[attr])
        velocity[attr] = inertia + cognitive + social
    return velocity

def run_pso_algorithm(data, iterations=50, population_size=30):
    """Runs the Particle Swarm Optimization algorithm on the given dataset."""
    particles = initialize_particles(data, population_size)
    personal_best = particles.copy()  # List to track personal bests for each particle
    global_best = max(personal_best, key=lambda x: x['fitness'])  # Track the best overall

    fitness_trends = []
    for generation in range(iterations):
        for i, particle in enumerate(particles):
            # Calculate fitness for the current particle
            particle['fitness'] = fitness_cal(particle, data)
            # Update personal best if the current fitness is better
            if particle['fitness'] > personal_best[i]['fitness']:
                personal_best[i] = particle.copy()  # Update personal best for this particle
        # Find the best particle in personal_best
        current_best = max(personal_best, key=lambda x: x['fitness'])
        if current_best['fitness'] > global_best['fitness']:
            global_best = current_best.copy()  # Update global best

        # Update positions based on velocities
        for i, particle in enumerate(particles):
            velocity = update_velocity(particle, personal_best[i], global_best)
            particle['age'] = max(18, min(64, int(particle['age'] + velocity['age'])))
            particle['bmi'] = max(15.0, min(40.0, particle['bmi'] + velocity['bmi']))
            particle['children'] = max(0, min(5, int(particle['children'] + velocity['children'])))

        fitness_trends.append(global_best['fitness'])

    return {
        'best_solution': global_best,
        'generations': list(range(iterations)),
        'fitness_values': fitness_trends
    }
