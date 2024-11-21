import numpy as np
import cv2
import matplotlib.pyplot as plt

# Objective function: Otsu's Thresholding
def otsu_threshold(image, threshold):
    # Separate the pixels into two classes
    foreground = image[image >= threshold]
    background = image[image < threshold]

    # Calculate the weights for each class
    w0 = len(background) / image.size
    w1 = len(foreground) / image.size

    # Calculate the means of each class
    if len(background) == 0 or len(foreground) == 0:  # Avoid division by zero
        return np.inf

    mean0 = np.mean(background)
    mean1 = np.mean(foreground)

    # Calculate the between-class variance
    variance_between = w0 * w1 * (mean0 - mean1) ** 2
    return -variance_between  # Negative since we are maximizing this value

# Lévy flight for random walk
def levy_flight(Lambda):
    sigma1 = (np.math.gamma(1 + Lambda) * np.sin(np.pi * Lambda / 2) /
              (np.math.gamma((1 + Lambda) / 2) * Lambda * 2 ** ((Lambda - 1) / 2))) ** (1 / Lambda)
    sigma2 = 1
    u = np.random.normal(0, sigma1, size=1)
    v = np.random.normal(0, sigma2, size=1)
    step = u / abs(v) ** (1 / Lambda)
    return step[0]

# Cuckoo Search Algorithm
class CuckooSearch:
    def __init__(self, obj_function, image, num_nests=15, pa=0.25, num_iterations=50, lower_bound=0, upper_bound=255):
        self.obj_function = obj_function
        self.image = image
        self.num_nests = num_nests
        self.pa = pa
        self.num_iterations = num_iterations
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.nests = np.random.uniform(self.lower_bound, self.upper_bound, self.num_nests).astype(int)
        self.fitness = np.array([self.obj_function(self.image, nest) for nest in self.nests])

    def get_best_nest(self):
        min_fitness_idx = np.argmin(self.fitness)
        return self.nests[min_fitness_idx], self.fitness[min_fitness_idx]

    def levy_flight_move(self, cuckoo):
        step_size = levy_flight(1.5)
        new_nest = cuckoo + int(step_size * (cuckoo - self.nests[np.random.randint(0, self.num_nests)]))
        return np.clip(new_nest, self.lower_bound, self.upper_bound)

    def replace_worst_nests(self):
        for i in range(self.num_nests):
            if np.random.rand() < self.pa:
                self.nests[i] = np.random.randint(self.lower_bound, self.upper_bound)
                self.fitness[i] = self.obj_function(self.image, self.nests[i])

    def run(self):
        best_solution, best_fitness = self.get_best_nest()

        for iteration in range(self.num_iterations):
            for i in range(self.num_nests):
                new_nest = self.levy_flight_move(self.nests[i])
                new_fitness = self.obj_function(self.image, new_nest)

                if new_fitness < self.fitness[i]:
                    self.nests[i] = new_nest
                    self.fitness[i] = new_fitness

            current_best, current_best_fitness = self.get_best_nest()
            if current_best_fitness < best_fitness:
                best_solution, best_fitness = current_best, current_best_fitness

            self.replace_worst_nests()

        return best_solution

# Load the grayscale image
image = cv2.imread("/content/algo.jpg", cv2.IMREAD_GRAYSCALE)

# Run Cuckoo Search for optimal threshold
cs = CuckooSearch(otsu_threshold, image, num_nests=15, pa=0.25, num_iterations=50, lower_bound=0, upper_bound=255)
best_threshold = cs.run()

# Apply the threshold to get the binary image
_, binary_image = cv2.threshold(image, best_threshold, 255, cv2.THRESH_BINARY)

# Display the result
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.subplot(1, 2, 2)
plt.title(f"Binary Image (Threshold = {best_threshold})")
plt.imshow(binary_image, cmap='gray')
plt.show()
