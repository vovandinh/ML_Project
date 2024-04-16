
import numpy as np
class MultinomialLogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def _softmax(self, Z):
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    def _initialize_parameters(self, num_features, num_classes):
        self.weights = np.random.randn(num_features, num_classes)
        self.bias = np.zeros((1, num_classes))

    def _compute_loss(self, Y, Y_pred):
        num_examples = Y.shape[0]
        loss = -1 / num_examples * np.sum(Y * np.log(Y_pred))
        return loss

    def _compute_gradients(self, X, Y, Y_pred):
        num_examples = X.shape[0]
        dZ = Y_pred - Y
        dW = 1 / num_examples * np.dot(X.T, dZ)
        db = 1 / num_examples * np.sum(dZ, axis=0, keepdims=True)
        return dW, db

    def fit(self, X, Y):
        num_examples, num_features = X.shape
        num_classes = Y.shape[1]
        self._initialize_parameters(num_features, num_classes)

        for i in range(self.num_iterations):
            # Forward propagation
            Z = np.dot(X, self.weights) + self.bias
            Y_pred = self._softmax(Z)

            # Compute loss
            loss = self._compute_loss(Y, Y_pred)

            # Compute gradients
            dW, db = self._compute_gradients(X, Y, Y_pred)

            # Update parameters
            self.weights -= self.learning_rate * dW
            self.bias -= self.learning_rate * db

            # Print loss every 100 iterations
            if i % 100 == 0:
                print(f"Iteration {i}, Loss: {loss:.4f}")

    def predict(self, X):
        Z = np.dot(X, self.weights) + self.bias
        Y_pred = self._softmax(Z)
        return Y_pred