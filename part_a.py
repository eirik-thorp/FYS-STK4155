import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from matplotlib.ticker import MaxNLocator

from functions import FrankeFunction, OLS, predict, bootstrap, Ridge, Lasso

#Part a = Generated data set
#Part b = terrain data

np.random.seed(42)

#number of data points
m = 100

#generate data
x1 = np.random.rand(m,1).flatten()
y1 = np.random.rand(m,1).flatten()

X = np.sort(x1, axis=0)
Y = np.sort(y1, axis=0)
x, y = np.meshgrid(X,Y)

z = FrankeFunction(x, y, noise=True)

#plot function
fig = plt.figure(figsize=plt.figaspect(0.4))
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
ax.set_zlabel(r'$z$')
ax.set_title("Franke Function with noise")
ax.set_zlim(-0.10, 1.40)
surf = ax.plot_surface(x, y, z, rstride=1, cstride=1,
                cmap='inferno', edgecolor='none')

#perform fit
x1 = x.ravel()
y1 = y.ravel()
z1 = z.ravel()

degree = 5

beta = OLS(x1, y1, z1, degree)
z_ = predict(x1, y1, beta, degree)


'''
OLS method with polynomials up to 5th order

for degree in range(1,6):

    beta = OLS(x1, y1, z1, degree)
    z_ = predict(x1, y1, beta, degree)
    mse = np.mean( (z_ - z1)**2 )
    R2 = 1 - np.sum( (z_ - z1)**2 )/np.sum( (z1- np.mean(z1))**2 )

    print(mse, R2)
'''

'''

Test ridge and lasso, plot lambda vs. relative error and R2

lds = np.linspace(0,0.1,10)
degree = 5

error = np.zeros(10)
R2 = np.zeros(10)

for i, ld in enumerate(lds):

    beta = Lasso(x1, y1, z1, degree, ld)
    z_ = predict(x1, y1, beta, degree)
    error[i] = np.mean( (z_ - z1)**2 )
    R2[i] = 1 - np.sum( (z_ - z1)**2 )/np.sum( (z1- np.mean(z1))**2 )

ax = plt.figure().gca()
plt.plot(lds, error/np.max(error), 'r-',label='Error')
plt.plot(lds, R2, 'g-',label='R2')
plt.xlabel('lambda')
plt.title('Lasso regression')
plt.legend()
plt.show()
'''

'''

Bootstrap, up to given degree

degrees = 5

error = np.zeros(degrees)
R2 = np.zeros(degrees)
bias = np.zeros(degrees)
variance = np.zeros(degrees)

for i in range (degrees):
    error[i], R2[i], bias[i], variance[i] = bootstrap(x1, y1, z1, degree=i+1)

ax = plt.figure().gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.plot(np.arange(1,degrees+1), bias/np.max(bias), 'r-.',label='bias')
plt.plot(np.arange(1,degrees+1), variance, 'g-',label='variance')
plt.plot(np.arange(1,degrees+1), error/np.max(error), 'b*',label='error')
plt.plot(np.arange(1,degrees+1), R2, 'y-',label='R2')
plt.xlabel('Degree of polynomials')
plt.title('Bootstrap using OLS regression')
plt.legend()
plt.show()

'''

#plot fit

z_ = z_.reshape(m,m)

ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.set_title("OLS fit with degree=5")
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
ax.set_zlabel(r'$z$')
surf = ax.plot_surface(x, y, z_, rstride=1, cstride=1,
                cmap='inferno', edgecolor='none')

ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.35, aspect=10)
plt.show()

#Error metrics
z_ = z_.ravel()
print("Mean Squared Error:", mean_squared_error(z1, z_))
print("R2 Score", r2_score(z1, z_))


'''
Confidence intervals of beta-parameters of OLS fit with degree up to 5th order


z_ = z_.ravel()
z = z.ravel()

num_data = m*m

x=x1
y=y1

xb = np.c_[np.ones((x.shape[0],1)), x, y, (x**2), (y**2), (x*y), (x**3), \
(y**3), (x*(y**2)), (y*(x**2)), (x**4), (y**4), (y*(x**3)), (x*(y**3)), \
((x**2)*(y**2)), (x**5), (y**5), (x*(y**4)), (y*(x**4)), ((x**2)*(y**3)), ((x**3)*(y**2))]

var2 = np.sum( (z_ - z)**2 )/ (num_data - 22)
H = np.linalg.inv(xb.T.dot(xb))
var_cov = H*var2
var = np.diagonal(var_cov)
np.set_printoptions(precision=5, suppress=True)

ci = np.zeros((21,2))
for i in range(21):
    ci[i, 0] = beta[i] - 1.96*(np.sqrt(var[i]))
    ci[i, 1] = beta[i] + 1.96*(np.sqrt(var[i]))
print(ci)
'''
