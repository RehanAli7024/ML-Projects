"""
Support Vector Machine (SVM) Implementation from Scratch
Author: Binary Classification Expert
Description: A complete implementation of SVM with different kernels using gradient descent
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_circles, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class SVM:
    """
    Support Vector Machine implementation using gradient descent
    """
    
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iterations=1000, kernel='linear', gamma=1.0, degree=3, coef0=0.0):
        """
        Initialize the SVM model
        
        Parameters:
        -----------
        learning_rate : float, default=0.001
            The learning rate for gradient descent
        lambda_param : float, default=0.01
            Regularization parameter (C = 1/lambda_param)
        n_iterations : int, default=1000
            Number of iterations for training
        kernel : str, default='linear'
            Kernel type ('linear', 'polynomial', 'rbf')
        gamma : float, default=1.0
            Kernel coefficient for 'rbf' and 'polynomial'
        degree : int, default=3
            Degree for polynomial kernel
        coef0 : float, default=0.0
            Independent term in polynomial kernel
        """
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iterations = n_iterations
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.w = None
        self.b = None
        self.X_train = None
        self.y_train = None
        self.alphas = None
        self.cost_history = []
        
    def linear_kernel(self, x1, x2):
        """Linear kernel function"""
        return np.dot(x1, x2.T)
    
    def polynomial_kernel(self, x1, x2):
        """Polynomial kernel function"""
        return (self.gamma * np.dot(x1, x2.T) + self.coef0) ** self.degree
    
    def rbf_kernel(self, x1, x2):
        """Radial Basis Function (Gaussian) kernel"""
        if x1.ndim == 1:
            x1 = x1.reshape(1, -1)
        if x2.ndim == 1:
            x2 = x2.reshape(1, -1)
            
        # Compute squared Euclidean distances
        sq_dists = np.sum(x1**2, axis=1).reshape(-1, 1) + np.sum(x2**2, axis=1) - 2 * np.dot(x1, x2.T)
        return np.exp(-self.gamma * sq_dists)
    
    def compute_kernel_matrix(self, X1, X2=None):
        """Compute kernel matrix between X1 and X2"""
        if X2 is None:
            X2 = X1
            
        if self.kernel == 'linear':
            return self.linear_kernel(X1, X2)
        elif self.kernel == 'polynomial':
            return self.polynomial_kernel(X1, X2)
        elif self.kernel == 'rbf':
            return self.rbf_kernel(X1, X2)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
    
    def fit(self, X, y):
        """
        Train the SVM model
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values (-1 or 1)
        """
        # Convert to numpy arrays and ensure y is in {-1, 1}
        X = np.array(X)
        y = np.array(y)
        
        # Convert labels to {-1, 1} if they are in {0, 1}
        if np.min(y) == 0:
            y = 2 * y - 1
            
        n_samples, n_features = X.shape
        
        # Store training data for kernel methods
        self.X_train = X
        self.y_train = y
        
        if self.kernel == 'linear':
            # Linear SVM using primal formulation
            self.w = np.zeros(n_features)
            self.b = 0
            
            for iteration in range(self.n_iterations):
                # Compute predictions
                scores = np.dot(X, self.w) + self.b
                
                # Compute hinge loss and cost
                margins = y * scores
                hinge_loss = np.maximum(0, 1 - margins)
                cost = self.lambda_param * np.dot(self.w, self.w) + np.mean(hinge_loss)
                self.cost_history.append(cost)
                
                # Compute gradients
                dw = 2 * self.lambda_param * self.w
                db = 0
                
                # Update gradients for misclassified points
                for i in range(n_samples):
                    if margins[i] < 1:  # Misclassified or in margin
                        dw -= y[i] * X[i] / n_samples
                        db -= y[i] / n_samples
                
                # Update parameters
                self.w -= self.learning_rate * dw
                self.b -= self.learning_rate * db
                
        else:
            # Kernel SVM using dual formulation (simplified)
            self.alphas = np.zeros(n_samples)
            self.b = 0
            
            # Compute kernel matrix
            K = self.compute_kernel_matrix(X, X)
            
            for iteration in range(self.n_iterations):
                # Simplified SMO-like updates
                for i in range(n_samples):
                    # Compute prediction for sample i
                    prediction = np.sum(self.alphas * y * K[i, :]) + self.b
                    
                    # Check KKT conditions and update alpha
                    if y[i] * prediction < 1:
                        self.alphas[i] += self.learning_rate
                    
                    # Clip alpha to [0, C]
                    self.alphas[i] = np.clip(self.alphas[i], 0, 1/self.lambda_param)
                
                # Update bias term
                support_vectors = self.alphas > 1e-5
                if np.sum(support_vectors) > 0:
                    self.b = np.mean(y[support_vectors] - np.sum(self.alphas * y * K[support_vectors, :], axis=1))
                
                # Compute cost (approximation)
                predictions = np.sum(self.alphas * y * K, axis=1) + self.b
                margins = y * predictions
                hinge_loss = np.maximum(0, 1 - margins)
                cost = 0.5 * np.sum(self.alphas * y * np.sum(self.alphas * y * K, axis=1)) + np.mean(hinge_loss)
                self.cost_history.append(cost)
    
    def predict(self, X):
        """
        Make predictions
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data
            
        Returns:
        --------
        array-like, shape (n_samples,)
            Predicted labels (-1 or 1)
        """
        X = np.array(X)
        
        if self.kernel == 'linear':
            scores = np.dot(X, self.w) + self.b
        else:
            # Kernel prediction
            K = self.compute_kernel_matrix(X, self.X_train)
            scores = np.sum(self.alphas * self.y_train * K, axis=1) + self.b
        
        return np.sign(scores)
    
    def decision_function(self, X):
        """
        Compute the decision function values
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data
            
        Returns:
        --------
        array-like, shape (n_samples,)
            Decision function values
        """
        X = np.array(X)
        
        if self.kernel == 'linear':
            return np.dot(X, self.w) + self.b
        else:
            K = self.compute_kernel_matrix(X, self.X_train)
            return np.sum(self.alphas * self.y_train * K, axis=1) + self.b
    
    def plot_cost_history(self):
        """Plot the cost function over iterations"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.cost_history)
        plt.title('SVM Cost Function Over Iterations')
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.grid(True)
        plt.show()
    
    def plot_decision_boundary(self, X, y, title="SVM Decision Boundary"):
        """
        Plot decision boundary for 2D data
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, 2)
            Input data (2D only)
        y : array-like, shape (n_samples,)
            Target labels
        title : str
            Plot title
        """
        if X.shape[1] != 2:
            print("Can only plot decision boundary for 2D data")
            return
            
        plt.figure(figsize=(10, 8))
        
        # Create a mesh
        h = 0.02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        # Make predictions on the mesh
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        Z = self.decision_function(mesh_points)
        Z = Z.reshape(xx.shape)
        
        # Plot the contour and training examples
        plt.contourf(xx, yy, Z, levels=50, alpha=0.8, cmap=plt.cm.RdYlBu)
        plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')
        
        # Plot the points
        scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='black')
        plt.colorbar(scatter)
        plt.title(title)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.show()


def accuracy_score(y_true, y_pred):
    """Calculate accuracy score"""
    return np.mean(y_true == y_pred)


def main():
    """
    Example usage of the SVM implementation with different kernels
    """
    print("Support Vector Machine Implementation Demo")
    print("=" * 50)
    
    # Test different datasets and kernels
    datasets = [
        ("Linearly Separable", make_classification(n_samples=200, n_features=2, n_redundant=0, 
                                                  n_informative=2, n_clusters_per_class=1, random_state=42)),
        ("Circular Data", make_circles(n_samples=200, noise=0.1, factor=0.5, random_state=42)),
        ("Moon Data", make_moons(n_samples=200, noise=0.1, random_state=42))
    ]
    
    kernels = ['linear', 'polynomial', 'rbf']
    
    for dataset_name, (X, y) in datasets:
        print(f"\n{dataset_name} Dataset:")
        print("-" * 30)
        
        # Convert labels to {-1, 1}
        y = 2 * y - 1
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_scaled = scaler.transform(X)
        
        for kernel in kernels:
            print(f"\n  {kernel.upper()} Kernel:")
            
            # Create and train the model
            if kernel == 'linear':
                svm = SVM(learning_rate=0.01, lambda_param=0.01, n_iterations=1000, kernel=kernel)
            elif kernel == 'polynomial':
                svm = SVM(learning_rate=0.001, lambda_param=0.01, n_iterations=1000, 
                         kernel=kernel, degree=3, gamma=1.0)
            else:  # rbf
                svm = SVM(learning_rate=0.001, lambda_param=0.01, n_iterations=1000, 
                         kernel=kernel, gamma=1.0)
            
            # Train the model
            svm.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_train_pred = svm.predict(X_train_scaled)
            y_test_pred = svm.predict(X_test_scaled)
            
            # Calculate accuracies
            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            
            print(f"    Training Accuracy: {train_accuracy:.4f}")
            print(f"    Test Accuracy: {test_accuracy:.4f}")
            
            # Plot decision boundary for the best performing kernel on each dataset
            if (dataset_name == "Linearly Separable" and kernel == 'linear') or \
               (dataset_name == "Circular Data" and kernel == 'rbf') or \
               (dataset_name == "Moon Data" and kernel == 'rbf'):
                svm.plot_decision_boundary(X_scaled, y, 
                                         title=f"{dataset_name} - {kernel.upper()} Kernel")
    
    # Demonstrate cost history plotting
    print("\nTraining Linear SVM and showing cost history:")
    X, y = make_classification(n_samples=300, n_features=2, n_redundant=0, 
                              n_informative=2, n_clusters_per_class=1, random_state=42)
    y = 2 * y - 1  # Convert to {-1, 1}
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    svm = SVM(learning_rate=0.01, lambda_param=0.01, n_iterations=500, kernel='linear')
    svm.fit(X_scaled, y)
    svm.plot_cost_history()


if __name__ == "__main__":
    main()
