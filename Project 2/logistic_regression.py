import numpy as np

class logistic_regression:

    def __init__(self, eta, num_iter, ld):

        self.eta = eta #learning rate
        self.num_iter = num_iter
        self.ld = ld #Lambda regularization parameter

    #add column of ones to the data-matrix
    def add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.hstack((intercept, X))

    def sigmoid(self, h):

        y = 1 / (1 + np.exp(-h))

        return y

    #Train for specified number of iterations with regularization
    def fit(self, X, true):

        X = self.add_intercept(X)
        self.weights = np.zeros(X.shape[1])

        for i in range(self.num_iter):

            pred = self.sigmoid(X @ self.weights)
            m = true.size
            gradient = (X.T @ (pred - true) + self.ld*np.sum(self.weights)) / m
            self.weights -= self.eta * gradient

    def predict(self, X):

        X = self.add_intercept(X)
        pred = self.sigmoid(X @ self.weights)

        return (pred >= 0.5)

    #Accuracy score
    def score(self, test, true):

        pred = self.predict(test)

        return (pred == true).mean()

    def get_coeff(self):

        return self.weights
