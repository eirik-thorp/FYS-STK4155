import numpy as np
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

import mlp
from functions import OLS, Ridge, Lasso, bootstrap, predict, plot_data

import warnings
#Comment this to turn on warnings
warnings.filterwarnings('ignore')

np.random.seed(12)

### define Ising model parameters
# system size
L=40

# create 10000 random Ising states
states=np.random.choice([-1, 1], size=(10000,L))

def ising_energies(states,L):
    """
    This function calculates the energies of the states in the nn Ising Hamiltonian
    """
    J=np.zeros((L,L),)
    for i in range(L):
        J[i,(i+1)%L]-=1.0
    # compute energies

    E = np.einsum('...i,ij,...j->...',states,J,states)
    return E

# calculate Ising energies
energies=ising_energies(states,L)

states=np.einsum('...i,...j->...ij', states, states)
shape=states.shape
states=states.reshape((shape[0],shape[1]*shape[2]))

Data=[states,energies]

x_train, x_test, y_train, y_test = train_test_split(Data[0], Data[1], test_size=0.20)

# Data as used by Mehta et al.
'''
# define number of samples
n_samples=400
# define train and test data sets
x_train=Data[0][:n_samples]
y_train=Data[1][:n_samples] #+ np.random.normal(0,4.0,size=X_train.shape[0])
x_test=Data[0][n_samples:3*n_samples//2]
y_test=Data[1][n_samples:3*n_samples//2] #+ np.random.normal(0,4.0,size=X_test.shape[0])
'''

lambdas = np.logspace(-4, 5, 10)

train_errors_leastsq = []
test_errors_leastsq = []

train_errors_ridge = []
test_errors_ridge = []

train_errors_lasso = []
test_errors_lasso = []

for ld in lambdas:

    beta = Ridge(x_train, y_train, degree=1, ld=0.00000001)
    test_pred = predict(x_test, beta, degree=1)
    train_pred = predict(x_train, beta, degree=1)

    test_errors_leastsq.append(r2_score(y_test, test_pred))
    train_errors_leastsq.append(r2_score(y_train, train_pred))

    beta = Ridge(x_train, y_train, degree=1, ld=ld)
    test_pred = predict(x_test, beta, degree=1)
    train_pred = predict(x_train, beta, degree=1)

    test_errors_ridge.append(r2_score(y_test, test_pred))
    train_errors_ridge.append(r2_score(y_train, train_pred))

    beta = Lasso(x_train, y_train, degree=1, ld=ld)
    test_pred = predict(x_test, beta, degree=1)
    train_pred = predict(x_train, beta, degree=1)

    test_errors_lasso.append(r2_score(y_test, test_pred))
    train_errors_lasso.append(r2_score(y_train, train_pred))

# Plot our performance on both the training and test data
plt.semilogx(lambdas, train_errors_leastsq, 'b',label='Train (OLS)')
plt.semilogx(lambdas, test_errors_leastsq,'--b',label='Test (OLS)')
plt.semilogx(lambdas, train_errors_ridge,'r',label='Train (Ridge)',linewidth=1)
plt.semilogx(lambdas, test_errors_ridge,'--r',label='Test (Ridge)',linewidth=1)
plt.semilogx(lambdas, train_errors_lasso, 'g',label='Train (LASSO)')
plt.semilogx(lambdas, test_errors_lasso, '--g',label='Test (LASSO)')

fig = plt.gcf()
plt.legend(loc='lower left',fontsize=9)
plt.ylim([-0.01, 1.01])
plt.xlim([min(lambdas), max(lambdas)])
plt.xlabel(r'$\lambda$',fontsize=12)
plt.ylabel('Performance',fontsize=12)
plt.tick_params(labelsize=8)
plt.show()

'''
Now using MLP
'''

size = y_train.shape[0]
y_train = y_train.reshape(size, -1)

size = y_test.shape[0]
y_test = y_test.reshape(size, -1)

#Values for grid search
nh = np.array([2,4,8,10]) #Number of nodes in hidden layer
et = np.array([0.001, 0.01, 0.1]) #Learning rate

train_accuracy = np.zeros((len(nh), len(et)), dtype=np.float64)
test_accuracy = np.zeros((len(nh), len(et)), dtype=np.float64)

for i, n in enumerate(nh):
    for j, e in enumerate(et):

        mlp = mlp.mlp(x_train, y_train, nhidden = n, eta=e, linear=True)
        mlp.earlystopping(x_train, y_train, x_test, y_test)

        preds_train = []
        preds_test = []

        for k in x_train:
            pred = mlp.forward(k)
            preds_train.append(pred)

        for k in x_test:
            pred = mlp.forward(k)
            preds_test.append(pred)

        train_accuracy[i,j] = r2_score(y_train, preds_train)
        test_accuracy[i,j] = r2_score(y_test, preds_test)

plot_data(et, nh, train_accuracy)
plot_data(et, nh, test_accuracy)
