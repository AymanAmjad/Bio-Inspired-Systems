import numpy as np
import cv2
import matplotlib.pyplot as plt

# Function to load and preprocess the image
def load_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load image in grayscale
    return image

# Function to calculate the fitness (difference between pixel and neighbors)
def calculate_fitness(grid):
    # Fitness can be based on the difference between a pixel and its neighbors (for smoothing)
    return grid  # For simplicity, fitness in this case is just the pixel value itself

# Function to update the grid based on neighborhood average (smoothing)
def update_grid(grid):
    new_grid = np.copy(grid)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            # Define the 3x3 neighborhood
            neighbors = []
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < grid.shape[0] and 0 <= nj < grid.shape[1]:  # Check bounds
                        neighbors.append(grid[ni, nj])
            # Update the pixel's value to the average of its neighbors
            new_grid[i, j] = np.mean(neighbors)
    return new_grid

# Function to denoise the image using Parallel Cellular Algorithm
def denoise_image(image, iterations=10):
    grid = np.copy(image)  # Initialize grid with the original image
    for _ in range(iterations):
        grid = update_grid(grid)  # Update the image based on neighbor averages
    return grid

# Load the noisy image
image_path = '/content/bis2.png'  # Path to the noisy image
image = load_image(image_path)

# Apply the Parallel Cellular Algorithm for image denoising
denoised_image = denoise_image(image, iterations=10)

# Display the original and denoised images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Denoised Image")
plt.imshow(denoised_image, cmap='gray')
plt.axis('off')

plt.show()

# Optionally, save the denoised image
cv2.imwrite('denoised_image.jpg', denoised_image)
