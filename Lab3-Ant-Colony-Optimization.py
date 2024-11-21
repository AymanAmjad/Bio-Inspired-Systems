import numpy as np
import random

class ACO_TSP:
    def __init__(self, cities, num_ants, alpha, beta, rho, initial_pheromone, num_iterations):
        self.cities = cities
        self.num_ants = num_ants
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.initial_pheromone = initial_pheromone
        self.num_iterations = num_iterations
        self.num_cities = len(cities)
        self.pheromone_matrix = np.full((self.num_cities, self.num_cities), initial_pheromone, dtype=float)
        self.distance_matrix = self.calculate_distances()
        self.best_path = None
        self.best_length = float('inf')

    def calculate_distances(self):
        """Calculate Euclidean distance between each pair of cities and print the distances."""
        dist_matrix = np.zeros((self.num_cities, self.num_cities))

        print("Distances between each pair of cities:")
        for i in range(self.num_cities):
            for j in range(self.num_cities):
                if i != j:
                    dist_matrix[i][j] = np.linalg.norm(np.array(self.cities[i]) - np.array(self.cities[j]))
                    print(f"Distance between City {i} and City {j}: {dist_matrix[i][j]:.2f}")
                else:
                    dist_matrix[i][j] = 0  # Distance to itself is zero
        print("\n")
        return dist_matrix
    def construct_solution(self):
            """Construct a path for each ant, starting from city 0, based on pheromone and heuristic information."""
            all_paths = []
            for ant in range(self.num_ants):
                unvisited = list(range(1, self.num_cities))  # Start with all cities except city 0
                path = [0]  # Start each ant's path from city 0

                while unvisited:
                    current_city = path[-1]
                    probabilities = []
                    for next_city in unvisited:
                        prob = (self.pheromone_matrix[current_city][next_city] ** self.alpha) * \
                               ((1 / self.distance_matrix[current_city][next_city]) ** self.beta)
                        probabilities.append(prob)
                    probabilities = np.array(probabilities)
                    probabilities /= probabilities.sum()  # Normalize to make it a probability distribution
                    next_city = np.random.choice(unvisited, p=probabilities)
                    path.append(next_city)
                    unvisited.remove(next_city)

                path.append(0)  # Return to starting city
                all_paths.append(path)
            return all_paths


    def path_length(self, path):
        """Calculate total distance of a path."""
        length = sum(self.distance_matrix[path[i]][path[i + 1]] for i in range(len(path) - 1))
        return length

    def update_pheromones(self, paths):
        """Evaporate and deposit pheromones based on path quality."""
        self.pheromone_matrix *= (1 - self.rho)  # Evaporation

        for path in paths:
            length = self.path_length(path)
            pheromone_deposit = 1 / length
            for i in range(len(path) - 1):
                self.pheromone_matrix[path[i]][path[i + 1]] += pheromone_deposit
                self.pheromone_matrix[path[i + 1]][path[i]] += pheromone_deposit

    def run(self):
        """Execute the ACO algorithm over iterations."""
        for iteration in range(self.num_iterations):
            paths = self.construct_solution()
            lengths = [self.path_length(path) for path in paths]

            # Find the best solution in this iteration
            min_length = min(lengths)
            if min_length < self.best_length:
                self.best_length = min_length
                self.best_path = paths[lengths.index(min_length)]

            # Update pheromones based on the solutions found
            self.update_pheromones(paths)

        return self.best_path, self.best_length

# Example usage
cities = [(0, 0), (1, 5), (2, 3), (5, 2), (6, 6), (8, 3)]
aco = ACO_TSP(cities, num_ants=10, alpha=1, beta=5, rho=0.5, initial_pheromone=1, num_iterations=100)
best_path, best_length = aco.run()

print("Best path:", best_path)
print("Best path length:", best_length)
