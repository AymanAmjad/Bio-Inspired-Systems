import numpy as np

# Objective function to minimize
def objective_function(x):
    return sum(x**2)  # Example: Sphere function (sum of squares)

# Parameters
population_size = 20  # Number of genetic sequences
num_genes = 10  # Length of each genetic sequence
mutation_rate = 0.1  # Probability of mutation
crossover_rate = 0.7  # Probability of crossover
num_generations = 100  # Number of generations
bounds = (-10, 10)  # Bounds for gene expression

# Step 1: Initialize Population
def initialize_population(pop_size, num_genes, bounds):
    return np.random.uniform(bounds[0], bounds[1], (pop_size, num_genes))

# Step 2: Evaluate Fitness
def evaluate_fitness(population):
    return np.array([objective_function(individual) for individual in population])

# Step 3: Selection (Tournament Selection)
def selection(population, fitness, tournament_size=3):
    selected = []
    for _ in range(len(population)):
        participants = np.random.choice(len(population), tournament_size)
        winner = participants[np.argmin(fitness[participants])]
        selected.append(population[winner])
    return np.array(selected)

# Step 4: Crossover (Single-Point Crossover)
def crossover(parent1, parent2):
    if np.random.rand() < crossover_rate:
        point = np.random.randint(1, len(parent1))
        child1 = np.concatenate((parent1[:point], parent2[point:]))
        child2 = np.concatenate((parent2[:point], parent1[point:]))
        return child1, child2
    return parent1.copy(), parent2.copy()

# Step 5: Mutation
def mutate(individual, mutation_rate, bounds):
    for i in range(len(individual)):
        if np.random.rand() < mutation_rate:
            individual[i] = np.random.uniform(bounds[0], bounds[1])
    return individual

# Main Optimization Loop
population = initialize_population(population_size, num_genes, bounds)
best_solution = None
best_fitness = float("inf")

for generation in range(num_generations):
    # Evaluate fitness
    fitness = evaluate_fitness(population)

    # Track the best solution
    min_index = np.argmin(fitness)
    if fitness[min_index] < best_fitness:
        best_fitness = fitness[min_index]
        best_solution = population[min_index]

    # Selection
    selected_population = selection(population, fitness)

    # Crossover and Mutation
    next_population = []
    for i in range(0, population_size, 2):
        parent1, parent2 = selected_population[i], selected_population[i + 1]
        child1, child2 = crossover(parent1, parent2)
        child1 = mutate(child1, mutation_rate, bounds)
        child2 = mutate(child2, mutation_rate, bounds)
        next_population.append(child1)
        next_population.append(child2)

    population = np.array(next_population)

# Output the Best Solution
print("Best solution:", *best_solution, sep="\n")
print(f"\nBest fitness: {best_fitness:.3f}")
