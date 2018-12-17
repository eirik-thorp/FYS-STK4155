import numpy as np
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

#x = independent variables, y = dependent variable, degree = max degree of polynomial
def OLS(x, y, degree):

    poly = PolynomialFeatures(degree)
    xb = poly.fit_transform(x)

    beta = np.linalg.inv(xb.T.dot(xb)).dot(xb.T).dot(z)

    return beta

#x = independent variables, y = dependent variable, degree = max degree of polynomial
#ld = constraint parameter
def Ridge(x, y, degree, ld):

    poly = PolynomialFeatures(degree)
    xb = poly.fit_transform(x)

    beta= np.linalg.inv((xb.T.dot(xb)) + ld*(np.identity(xb.shape[1]))).dot(xb.T).dot(y)

    return beta

#x = independent variables, y = dependent variable, degree = max degree of polynomial
#ld = constraint parameter
def Lasso(x, y, degree, ld):

    a = PolynomialFeatures(degree)
    xb = a.fit_transform(x)

    lasso=linear_model.Lasso(alpha=ld, fit_intercept=False, max_iter=10000)
    lasso.fit(xb,y)

    beta = lasso.coef_

    return beta

#predict new values using estimated beta parameters
def predict(x, beta, degree):

    poly = PolynomialFeatures(degree)
    terms = poly.fit_transform(x)

    return terms @ beta

# Non-parametric bootstrap, see report for more detail
def bootstrap(x, y, degree):

    #Split data into test and train
    x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.2)

    #number of bootstrap iterations
    n_boostraps = 100

    #Store all predicted
    y_pred = np.empty((y_test.shape[0], n_boostraps))

    for i in range(n_boostraps):
        x_, y_ = resample(x_train, y_train)
        beta = Ridge(x_, y_, degree, 0.0001)
        y_pred[:, i] = predict(x_test, beta, degree)

    y_test = y_test.reshape(-1,1)

    error = np.mean( np.mean((y_test - y_pred)**2, axis=1, keepdims=True) )
    R2 = 1 - np.sum( np.mean((y_test - y_pred)**2, axis=1, keepdims=True) ) \
    / np.sum( (y_test- np.mean(y_test))**2 )
    bias = np.mean( (y_test - np.mean(y_pred, axis=1, keepdims=True))**2 )
    variance = np.mean( np.var(y_pred, axis=1, keepdims=True) )

    return error, R2, bias, variance

def plot_data(x,y,data):

    # plot results
    fontsize=16


    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(data, interpolation='nearest', vmin=0, vmax=1)
    fig.colorbar(cax)

    # put text on matrix elements
    for i, x_val in enumerate(np.arange(len(x))):
        for j, y_val in enumerate(np.arange(len(y))):
            c = "${0:.1f}\\%$".format( 100*data[j,i])
            ax.text(x_val, y_val, c, va='center', ha='center')

    # convert axis values to to string labels
    x=[str(i) for i in x]
    y=[str(i) for i in y]


    ax.set_xticklabels(['']+x)
    ax.set_yticklabels(['']+y)

    ax.set_xlabel('$\\mathrm{learning\\ rate}$',fontsize=fontsize)
    ax.set_ylabel('$\\mathrm{Hidden nodes}$',fontsize=fontsize)

    plt.tight_layout()

    plt.title('MLP on 2nd matrix')

    plt.show()
