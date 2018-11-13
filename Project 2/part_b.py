import pickle
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.model_selection as skms
import sklearn.linear_model as skl
import sklearn.metrics as skm
import sklearn.preprocessing as pre
from keras.utils import to_categorical

from functions import plot_data

import logistic_regression
import mlp

label_filename = 'Ising2DFM_reSample_L40_T=All_labels.pkl'
dat_filename = 'Ising2DFM_reSample_L40_T=All.pkl'

# Read in the labels
with open(label_filename, "rb") as f:
    labels = pickle.load(f)

# Read in the corresponding configurations
with open(dat_filename, "rb") as f:
    data = np.unpackbits(pickle.load(f)).reshape(-1, 1600).astype("int")

# Set spin-down to -1
#data[data == 0] = -1

ordered = slice(0, 70000)
critical = slice(70000, 100000)
disordered = slice(100000, 160000)

X_train, X_test, y_train, y_test = skms.train_test_split(
    np.concatenate((data[ordered], data[disordered])),
    np.concatenate((labels[ordered], labels[disordered])),
    test_size=0.20)

data_critical = np.concatenate((data[critical], data[critical]))
labels_critical = np.concatenate((labels[critical], labels[critical]))

'''
Classification using logistic regression,
A grid-search is used to find the optimal paramters
'''

et = np.array([0.001, 0.01, 0.1]) #Learning rate
ld = np.array([0.01, 0.1, 1]) #Lambda regularization paramater

train_accuracy = np.zeros((len(et), len(ld)))
test_accuracy = np.zeros((len(et), len(ld)))
critical_accuracy = np.zeros((len(et), len(ld)))

for i,l in enumerate(ld):
    for j,e in enumerate(et):

        clf = logistic_regression.logistic_regression(eta=e, num_iter=1000, ld=l)
        clf.fit(X_train, y_train)

        train_accuracy[i,j] = (clf.score(X_train, y_train))
        test_accuracy[i,j] = (clf.score(X_test, y_test))
        critical_accuracy[i,j] = (clf.score(data_critical, labels_critical))

plot_data(et, ld, train_accuracy)
plot_data(et, ld, test_accuracy)
plot_data(et, ld, critical_accuracy)

'''
Now using MLP
'''

#Turn into one-hot vector
y_test = to_categorical(y_test)
y_train = to_categorical(y_train)
labels_critical = to_categorical(labels_critical)

#parameter values
nh = np.array([50, 100]) #hidden nodes
et = np.array([0.01, 0.1]) #Learning rate

train_accuracy = np.zeros((len(nh), len(et)))
test_accuracy = np.zeros((len(nh), len(et)))
critical_accuracy = np.zeros((len(nh), len(et)))

for i, n in enumerate(nh):
    for j, e in enumerate(et):

        mlp = mlp.mlp(X_train, y_train, nhidden=n, eta=e, linear=False)
        mlp.earlystopping(X_train, y_train, X_test, y_test)
        train_accuracy[i,j] = (mlp.score(X_train, y_train))
        test_accuracy[i,j] = (mlp.score(X_test, y_test))
        critical_accuracy[i,j] = (mlp.score(data_critical, labels_critical))

plot_data(et, nh, train_accuracy)
plot_data(et, nh, test_accuracy)
plot_data(et, nh, critical_accuracy)
