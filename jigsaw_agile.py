"""
Logistic Regression Implementation from Scratch
Author: Jigsaw Agile
Description: A complete implementation of logistic regression using gradient descent
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class LogisticRegression:
    """
    Logistic Regression implementation using gradient descent
    """
    
    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
        """
        Initialize the Logistic Regression model
        
        Parameters:
        -----------
        learning_rate : float, default=0.01
            The learning rate for gradient descent
        max_iterations : int, default=1000
            Maximum number of iterations for gradient descent
        tolerance : float, default=1e-6
            Tolerance for convergence
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.weights = None
        self.bias = None
        self.cost_history = []
        
    def sigmoid(self, z):
        """
        Sigmoid activation function
        
        Parameters:
        -----------
        z : array-like
            Input values
            
        Returns:
        --------
        array-like
            Sigmoid of input values
        """
        # Clip z to prevent overflow
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))
    
    def compute_cost(self, y_true, y_pred):
        """
        Compute the logistic regression cost function (log-likelihood)
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted probabilities
            
        Returns:
        --------
        float
            Cost value
        """
        # Avoid log(0) by adding small epsilon
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        cost = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return cost
    
    def fit(self, X, y):
        """
        Train the logistic regression model
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values (0 or 1)
        """
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Get dimensions
        n_samples, n_features = X.shape
        
        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for i in range(self.max_iterations):
            # Forward pass
            linear_pred = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_pred)
            
            # Compute cost
            cost = self.compute_cost(y, y_pred)
            self.cost_history.append(cost)
            
            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Check for convergence
            if i > 0 and abs(self.cost_history[-2] - self.cost_history[-1]) < self.tolerance:
                print(f"Converged after {i+1} iterations")
                break
    
    def predict_proba(self, X):
        """
        Predict class probabilities
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data
            
        Returns:
        --------
        array-like, shape (n_samples,)
            Predicted probabilities
        """
        X = np.array(X)
        linear_pred = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_pred)
    
    def predict(self, X, threshold=0.5):
        """
        Make binary predictions
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data
        threshold : float, default=0.5
            Decision threshold
            
        Returns:
        --------
        array-like, shape (n_samples,)
            Binary predictions (0 or 1)
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
    
    def plot_cost_history(self):
        """
        Plot the cost function over iterations
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.cost_history)
        plt.title('Cost Function Over Iterations')
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.grid(True)
        plt.show()


def accuracy_score(y_true, y_pred):
    """
    Calculate accuracy score
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
        
    Returns:
    --------
    float
        Accuracy score
    """
    return np.mean(y_true == y_pred)


def confusion_matrix(y_true, y_pred):
    """
    Calculate confusion matrix
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
        
    Returns:
    --------
    array-like, shape (2, 2)
        Confusion matrix
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    return np.array([[tn, fp], [fn, tp]])


def main():
    """
    Example usage of the Logistic Regression implementation
    """
    print("Logistic Regression Implementation Demo")
    print("=" * 40)
    
    # Generate sample dataset
    X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, 
                             n_informative=2, n_clusters_per_class=1, 
                             random_state=42)
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                        random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train the model
    model = LogisticRegression(learning_rate=0.1, max_iterations=1000)
    print("Training the model...")
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Calculate accuracies
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"\nResults:")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Display confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    print(f"\nConfusion Matrix:")
    print(cm)
    
    # Plot cost history
    model.plot_cost_history()
    
    # Example of predicting probabilities
    probabilities = model.predict_proba(X_test_scaled[:5])
    print(f"\nSample Probabilities for first 5 test samples:")
    for i, prob in enumerate(probabilities):
        print(f"Sample {i+1}: {prob:.4f} (Predicted: {y_test_pred[i]}, Actual: {y_test[i]})")


if __name__ == "__main__":
    main()
