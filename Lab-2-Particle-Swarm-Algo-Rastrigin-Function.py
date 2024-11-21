import numpy as np
def rastrigin(X):
    return 10 * len(X) + sum([x**2 - 10 * np.cos(2 * np.pi * x) for x in X])

class Particle:
    def __init__(self, bounds):
        self.position = np.random.uniform(bounds[0], bounds[1], len(bounds[0]))
        self.velocity = np.random.uniform(-1, 1, len(bounds[0]))
        self.best_position = np.copy(self.position)
        self.best_value = rastrigin(self.position)
        self.current_value = self.best_value

    def update_velocity(self, global_best_position, w, c1, c2):
        inertia = w * self.velocity
        cognitive = c1 * np.random.random(len(self.position)) * (self.best_position - self.position)
        social = c2 * np.random.random(len(self.position)) * (global_best_position - self.position)
        self.velocity = inertia + cognitive + social

    def update_position(self, bounds):
        self.position += self.velocity
        self.position = np.clip(self.position, bounds[0], bounds[1])


    def evaluate(self):
        self.current_value = rastrigin(self.position)
        if self.current_value < self.best_value:
            self.best_value = self.current_value
            self.best_position = np.copy(self.position)

def pso(rastrigin_function, bounds, num_particles, max_iter):
    swarm = [Particle(bounds) for _ in range(num_particles)]
    global_best_position = None
    global_best_value = float('inf')

    for iteration in range(max_iter):
        for particle in swarm:
            particle.evaluate()

            if particle.best_value < global_best_value:
                global_best_value = particle.best_value
                global_best_position = np.copy(particle.best_position)

        for particle in swarm:
            particle.update_velocity(global_best_position, w=0.5, c1=1.5, c2=1.5)
            particle.update_position(bounds)
        print(f"Iteration {iteration+1}/{max_iter}, Best Value: {global_best_value}")

    return global_best_position, global_best_value

bounds = [(-5.12, -5.12), (5.12, 5.12)]
num_particles = 30
max_iter = 10

best_position, best_value = pso(rastrigin, bounds, num_particles, max_iter)

print("\nBest position found by PSO:", best_position)
print("Best value (minimized Rastrigin function):", best_value)
