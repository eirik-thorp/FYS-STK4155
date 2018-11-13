import numpy as np
import random
import math

np.random.seed(42)

class mlp:

    def __init__(self, inputs, targets, nhidden, eta, linear=False):

        self.eta = eta #learning rate

        self.linear = linear #activation on the output

        self.nhidden = nhidden #number of nodes in hidden layer
        self.n_in = inputs.shape[1] #Number of nodes in input layer
        self.n_out = targets.shape[1] #NUmber of nodes in output layer

        # First weight layer
        self.v = np.random.uniform(-0.7,0.7,((self.n_in, self.nhidden)))
        # Second weight layer
        self.w = np.random.uniform(-0.7,0.7,((self.nhidden, self.n_out)))
        #Hidden layer activation levels
        self.a = np.zeros(self.nhidden)

        #Hidden bias
        self.hidden_bias = np.zeros(self.nhidden) + 0.01

        #Output bias
        self.output_bias = np.zeros(self.n_out) + 0.01

    # Sigmoid activation function
    def sigmoid(self, h):

        y = 1 / (1 + np.exp(-h))

        return y

    # Derivative of the sigmoid function
    def delta_sigmoid(self, h):

        y = self.sigmoid(h) * (1 - self.sigmoid(h))

        return y

    # Error at the output
    def delta_output(self, out_pred, out_real):

        delta = (out_pred - out_real)

        if (not self.linear):
            delta = delta*self.delta_sigmoid(out_real)

        return delta

    # Error in the hidden layer
    def delta_hidden(self, output_error):

        delta = self.a*(1 - self.a) * (output_error @ self.w.T)
        return delta

    # See report for details
    def earlystopping(self, inputs, targets, valid, validtargets):

        n_epochs = 10
        prev_error = 1

        for i in range(n_epochs):

            self.train(inputs, targets)
            score = self.score(valid, validtargets)
            error = 1 - score

            error_change = (error - prev_error)

            if (i != 0 and error_change > 0):
                print('-- Earlystopping after %i epochs --' % (i+1))
                break
            else:
                prev_error = error

    #Training with random batches of size 100
    #Batches were not used for the regression problem
    def train(self, inputs, targets, iterations=10):

        n_train = inputs.shape[0]
        k = 100

        for i in range(iterations):

            #Shuffle data
            p = np.random.permutation(n_train)
            inputs, targets = inputs[p], targets[p]

            inp = inputs[0:k]
            tar = targets[0:k]

            n_batch = inp.shape[0]

            for j in range(n_batch):

                pred = self.forward(inp[j])
                true = tar[j]

                output_error = self.delta_output(pred, true)
                hidden_error = self.delta_hidden(output_error)

                output_bias_gradient = np.sum(output_error, axis=0)
                hidden_bias_gradient = np.sum(hidden_error, axis=0)

                updatew = np.ones(self.w.shape)
                updatew = (updatew * output_error).T * self.a * self.eta
                self.w -= updatew.T

                updatev = np.ones(self.v.shape)
                updatev = (updatev * hidden_error).T * inp[j,:] * self.eta
                self.v -= updatev.T

                self.output_bias -= self.eta * output_bias_gradient
                self.hidden_bias -= self.eta * hidden_bias_gradient

    #Feed the network forward
    def forward(self, inputs):

        # Calculate hidden layer
        self.a = self.sigmoid((self.v.T @ inputs) + self.hidden_bias)

        # Calculate output
        out = ((self.w.T @ self.a) + self.output_bias)

        if (not self.linear):
            out = self.sigmoid(out)

        return out

    #Optional confusion matrix
    def confusion(self, inputs, targets):

        n_data = inputs.shape[0]

        matrix = np.zeros((self.n_out, self.n_out))

        error = 0

        for i in range(n_data):

            pred = self.forward(inputs[i])
            true = targets[i]

            pred_class = np.argmax(pred)
            true_class = np.argmax(true)

            if (pred_class != true_class):
                error += 1

            matrix[pred_class][true_class] += 1

        prediction_rate = (n_data - error)/n_data

        print('Prediction rate: %f' %(prediction_rate))
        print('Confusion Matrix:')
        print('Rows - Predicted values\nColumns - True values')

        print(matrix)

        return prediction_rate

    #Accuracy score
    def score(self, inputs, targets):

        n = inputs.shape[0]

        score = 0.0

        for i in range(n):

            pred = self.forward(inputs[i])
            true = targets[i]

            pred_class = np.argmax(pred)
            true_class = np.argmax(true)

            if (pred_class == true_class):
                score += 1

        return score/n
