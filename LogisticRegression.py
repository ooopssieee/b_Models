import numpy as np

class LogisticRegression:
    def __init__(self, learningRate=0.03, epochs=10000, reg_param=0.1, decisionBoundary=0.5):
        self.learningRate = learningRate
        self.epochs = epochs
        self.regularizedParam = reg_param
        self.decisionBoundary = decisionBoundary
        self.weights = None
        self.bias = None 

    def fit(self, X, y):
        m, n = X.shape

        self.weights = np.zeros(n)
        self.bias = 0

        for _ in range(self.epochs):
            y_predicted = 1 / (1 + np.exp(-(np.dot(X, self.weights) + self.bias)))

            dw = (1 / m) * np.dot(X.T, (y_predicted - y)) + (self.regularizedParam / m) * self.weights
            db = (1 / m) * np.sum(y_predicted - y)

            self.weights -= self.learningRate * dw
            self.bias -= self.learningRate * db

    def predict(self, X):
        return np.where(1 / (1 + np.exp(-(np.dot(X, self.weights) + self.bias))) >= self.decisionBoundary, 1, 0)
