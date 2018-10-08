import numpy as np
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def FrankeFunction(x,y,noise=False):

    if (noise):
        noise_factor = 0.25
        noise = noise_factor*np.random.normal(0,1,x.shape[0])
    else:
        noise = 0

    term1 = 0.75*np.exp(-(0.25*(9*x+noise-2)**2) - 0.25*((9*y+noise-2)**2))
    term2 = 0.75*np.exp(-((9*x+noise+1)**2)/49.0 - 0.1*(9*y+noise+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y+noise-3)**2))
    term4 = -0.2*np.exp(-(9*x+noise-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4


#x, y = independent variables, z = dependent variable, degree = max degree of polynomial
def OLS(x, y, z, degree):

    X = np.column_stack((x,y))
    poly = PolynomialFeatures(degree)
    xb = poly.fit_transform(X)

    beta = np.linalg.inv(xb.T.dot(xb)).dot(xb.T).dot(z)

    return beta

#x, y = independent variables, z = dependent variable, degree = max degree of polynomial
#ld = constraint parameter
def Ridge(x, y, z, degree, ld):

    X = np.column_stack((x,y))
    poly = PolynomialFeatures(degree)
    xb = poly.fit_transform(X)

    beta= np.linalg.inv((xb.T.dot(xb)) + ld*(np.identity(xb.shape[1]))).dot(xb.T).dot(z)

    return beta

#x, y = independent variables, z = dependent variable, degree = max degree of polynomial
#ld = constraint parameter
def Lasso(x, y, z, degree, ld):

    xb = np.column_stack((x,y))
    a = PolynomialFeatures(degree)
    xb = a.fit_transform(xb)

    lasso=linear_model.Lasso(alpha=ld, fit_intercept=False, max_iter=10000)
    lasso.fit(xb,z)

    beta = lasso.coef_

    return beta

#predict new values using estimated beta parameters
def predict(x, y, beta, degree):

    X = np.column_stack((x,y))
    poly = PolynomialFeatures(degree)
    terms = poly.fit_transform(X)

    return terms @ beta

# Non-parametric bootstrap, see report for more detail

def bootstrap(x, y, z, degree):

    x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x, y, z, test_size=0.2)

    n_boostraps = 100
    z_pred = np.empty((z_test.shape[0], n_boostraps))

    for i in range(n_boostraps):
        x_, y_, z_ = resample(x_train, y_train, z_train)
        beta = OLS(x_, y_, z_, degree)
        z_pred[:, i] = predict(x_test, y_test, beta, degree)

    z_test = z_test.reshape(-1,1)

    error = np.mean( np.mean((z_test - z_pred)**2, axis=1, keepdims=True) )
    R2 = 1 - np.sum( np.mean((z_test - z_pred)**2, axis=1, keepdims=True) ) \
    / np.sum( (z_test- np.mean(z_test))**2 )
    bias = np.mean( (z_test - np.mean(z_pred, axis=1, keepdims=True))**2 )
    variance = np.mean( np.var(z_pred, axis=1, keepdims=True) )

    return error, R2, bias, variance
