import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from functions import OLS, predict, Ridge, Lasso, bootstrap

'''
Analysis of the digital terrain data

'''


#Read data and create terrain patch

terrain1 = imread('hawaii.tif')

row_start = 1000
row_end = 2500

col_start = 600
col_end = 3000

patch = terrain1[row_start:row_end, col_start:col_end]

rows = np.linspace(0,1,patch.shape[0])
cols = np.linspace(0,1,patch.shape[1])
[C,R] = np.meshgrid(cols,rows)

x = C.reshape(-1,1)
y = R.reshape(-1,1)

#Plot terrain patch
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(C,R,patch,cmap=cm.viridis,linewidth=0)
plt.title('Terrain patch')
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
ax.set_zlabel(r'$z$')
fig.colorbar(surf, shrink=0.35, aspect=10)
plt.show()
plt.show()

#Fit OLS, Ridge or Lasso, predict training data
x1 = x.ravel()
y1 = y.ravel()
z1 = patch.ravel()

degrees = 5
beta = OLS(x1, y1, z1, degrees) #Change to Lasso or Ridge
z_ = predict(x1, y1, beta, degrees)

'''
Test dependency of lambda on the ridge and lasso regression

lds = np.linspace(0,1,10)
degree = 5

error = np.zeros(10)
R2 = np.zeros(10)

for i, ld in enumerate(lds):

    beta = Ridge(x1, y1, z1, degree, ld)
    z_ = predict(x1, y1, beta, degree)
    error[i] = np.mean( (z_ - z1)**2 )
    R2[i] = 1 - np.sum( (z_ - z1)**2 )/np.sum( (z1- np.mean(z1))**2 )

ax = plt.figure().gca()
plt.plot(lds, error/np.max(error), 'r-',label='Relative error')
plt.plot(lds, R2, 'g-',label='R2')
plt.xlabel('lambda')
plt.title('Ridge regression on the terrain patch')
plt.legend()
plt.show()
'''

'''
#Bootstrap, with OLS used as default
#Can be changed to Ridge or Lasso in the functions.py file

error, R2, bias, variance = bootstrap(x1, y1, z1,degree=degrees)
print('Error:', error)
print('R2:', R2)
print('Bias^2:', bias)
print('Var:', variance)
'''

#Plot fit

z_ = z_.reshape(patch.shape[0],patch.shape[1])

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(C,R,z_,cmap=cm.viridis,linewidth=0)
plt.title('OLS fit')
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
ax.set_zlabel(r'$z$')
fig.colorbar(surf, shrink=0.35, aspect=10)
plt.show()

#Check MSE and R2 score
z_ = z_.ravel()

mse = np.mean( (z_ - z1)**2 )
R2 = 1 - np.sum( (z_ - z1)**2 )/np.sum( (z1- np.mean(z1))**2 )

#print('mse:', mse)
#print('R2:', R2)
