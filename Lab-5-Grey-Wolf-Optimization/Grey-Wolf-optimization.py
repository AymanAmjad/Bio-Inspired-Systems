import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Grey Wolf Optimizer (GWO) class
class GreyWolfOptimizer:
    def __init__(self, n_wolves, n_features, n_iterations, lower_bound, upper_bound, X_train, y_train):
        self.n_wolves = n_wolves
        self.n_features = n_features
        self.n_iterations = n_iterations
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.X_train = X_train
        self.y_train = y_train
        # Initialize wolves with random positions (feature selection probabilities between 0 and 1)
        self.position = np.random.uniform(self.lower_bound, self.upper_bound, (self.n_wolves, self.n_features))
        self.fitness = np.zeros(self.n_wolves)
        self.alpha_position = np.zeros(self.n_features)
        self.alpha_score = float("inf")
        self.beta_position = np.zeros(self.n_features)
        self.beta_score = float("inf")
        self.delta_position = np.zeros(self.n_features)
        self.delta_score = float("inf")

    def fitness_function(self, position):
        """Fitness function to evaluate classification accuracy based on feature selection"""
        selected_features = np.where(position > 0.5)[0]  # Features selected if > 0.5
        if len(selected_features) == 0:  # Avoid empty feature subset
            return float("inf")

        X_selected = self.X_train[:, selected_features]
        
        # Train a classifier (KNN in this case)
        classifier = KNeighborsClassifier(n_neighbors=3)
        classifier.fit(X_selected, self.y_train)
        
        # Predict and calculate accuracy
        accuracy = classifier.score(X_selected, self.y_train)
        return -accuracy  # We negate because we minimize in GWO (maximizing accuracy)

    def update_position(self, wolf_idx):
        """Update position of a wolf based on the alpha, beta, and delta wolves"""
        a = 2 - self.n_iterations * (2 / self.n_iterations)  # a decreases linearly
        a = max(a, 0.5)  # Prevent a from decaying too fast, keep some exploration
        r1, r2 = np.random.rand(2)
        A = 2 * a * r1 - a
        C = 2 * r2
        D_alpha = np.abs(C * self.alpha_position - self.position[wolf_idx])
        D_beta = np.abs(C * self.beta_position - self.position[wolf_idx])
        D_delta = np.abs(C * self.delta_position - self.position[wolf_idx])
        
        X1 = self.alpha_position - A * D_alpha
        X2 = self.beta_position - A * D_beta
        X3 = self.delta_position - A * D_delta
        self.position[wolf_idx] = (X1 + X2 + X3) / 3  # Update position based on the 3 best wolves
        
        # Ensure the position remains within bounds
        self.position[wolf_idx] = np.clip(self.position[wolf_idx], self.lower_bound, self.upper_bound)

    def run(self):
        for iteration in range(self.n_iterations):
            print(f"Iteration {iteration + 1}:")
            for i in range(self.n_wolves):
                # Calculate the fitness of each wolf (feature selection)
                self.fitness[i] = self.fitness_function(self.position[i])
                print(f"Wolf {i}, Fitness: {self.fitness[i]}")
                
                # Update alpha, beta, and delta wolves
                if self.fitness[i] < self.alpha_score:
                    self.alpha_score = self.fitness[i]
                    self.alpha_position = self.position[i]
                
                if self.fitness[i] < self.beta_score and self.fitness[i] != self.alpha_score:
                    self.beta_score = self.fitness[i]
                    self.beta_position = self.position[i]
                
                if self.fitness[i] < self.delta_score and self.fitness[i] != self.alpha_score and self.fitness[i] != self.beta_score:
                    self.delta_score = self.fitness[i]
                    self.delta_position = self.position[i]
            
            print(f"Alpha score: {self.alpha_score}")
            # Update positions of wolves
            for i in range(self.n_wolves):
                self.update_position(i)
        
        # After iterations, return the best feature subset found by alpha wolf
        selected_features = np.where(self.alpha_position > 0.5)[0]  # Extract selected features based on alpha's position
        
        # If no features are selected, select the top 5 features as a fallback
        if len(selected_features) == 0:
            print("No features selected. Falling back to top 5 features.")
            selected_features = np.argsort(self.alpha_position)[-5:]
        
        return selected_features

print("USN:1BM22CS061 NAME:AYMAN AMJAD")
# Load the Wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Hyperparameters for GWO
n_wolves = 30
n_features = X_train.shape[1]
n_iterations = 50
lower_bound = 0
upper_bound = 1

# Initialize and run GWO for feature selection
gwo = GreyWolfOptimizer(n_wolves, n_features, n_iterations, lower_bound, upper_bound, X_train, y_train)
best_features = gwo.run()

# Print selected features by GWO
print(f"Selected features: {best_features}")

# Evaluate the model with selected features
X_selected_train = X_train[:, best_features]
X_selected_test = X_test[:, best_features]

# Train KNN classifier with selected features
classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X_selected_train, y_train)
y_pred = classifier.predict(X_selected_test)

# Print the classification accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Classification accuracy with selected features: {accuracy * 100:.2f}%")
