import random

# Define the problem
def fitness_function(chromosome, weights, values, capacity):
    total_weight = sum(chromosome[i] * weights[i] for i in range(len(chromosome)))
    total_value = sum(chromosome[i] * values[i] for i in range(len(chromosome)))
    if total_weight > capacity:
        return -float("inf")  # Strictly exclude infeasible solutions
    return total_value

# Feasible Initialization
def initialize_population(pop_size, weights, capacity):
    population = []
    num_items = len(weights)
    while len(population) < pop_size:
        chromosome = [random.choice([0, 1]) for _ in range(num_items)]
        total_weight = sum(chromosome[i] * weights[i] for i in range(num_items))
        if total_weight <= capacity:
            population.append(chromosome)
    return population

# Selection: Tournament selection
def select_parents(population, fitnesses):
    valid_indices = [i for i, fit in enumerate(fitnesses) if fit != -float("inf")]
    tournament = random.sample(valid_indices, 3)
    tournament = sorted(tournament, key=lambda x: fitnesses[x], reverse=True)
    return population[tournament[0]], population[tournament[1]]

# Crossover: Single-point crossover
def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

# Mutation: Randomly flip a bit
def mutate(chromosome, mutation_rate):
    return [
        1 - gene if random.random() < mutation_rate else gene for gene in chromosome
    ]

# Main Genetic Algorithm
def genetic_algorithm(weights, values, capacity, pop_size=50, generations=25, mutation_rate=0.01):
    population = initialize_population(pop_size, weights, capacity)
    num_items = len(weights)

    for generation in range(generations):
        # Evaluate fitness of each individual
        fitnesses = [fitness_function(individual, weights, values, capacity) for individual in population]

        # Track the best solution
        best_index = fitnesses.index(max(fitnesses))
        best_solution = population[best_index]
        best_value = fitnesses[best_index]

        print(f"Generation {generation + 1}, Best Value: {best_value}")

        # Create next generation
        new_population = []
        while len(new_population) < pop_size:
            parent1, parent2 = select_parents(population, fitnesses)
            child1, child2 = crossover(parent1, parent2)
            new_population.extend([mutate(child1, mutation_rate), mutate(child2, mutation_rate)])

        # Ensure valid solutions
        population = initialize_population(pop_size, weights, capacity)

    # Return the best solution found
    return best_solution, best_value

# Problem setup
weights = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
values = [3, 4, 8, 8, 9, 10, 12, 14, 15, 16, 17, 18, 20, 22, 24, 26, 28, 30, 32, 34]
capacity = 50  # Maximum weight capacity

# Run the Genetic Algorithm
best_solution, best_value = genetic_algorithm(weights, values, capacity)

print("\nBest Solution (Items selected):", best_solution)
print("Best Value (Maximum profit):", best_value)
