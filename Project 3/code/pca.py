import numpy as np
import matplotlib.pyplot as plt

'''Principal Component Analysis'''

class pca:

    def __init__(self, n):

        self.n = n #Number of components or proportion of variance explained

        return

    #mean and standard deviation of training data
    def scale_fit(self, X):

        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)

        return

    #Standardize the data
    def scale_transform(self, X):

        Z = (X - self.mean) / self.std

        Z = np.nan_to_num(Z)

        return Z

    #Singular value decomposition of training data
    def pca_fit(self, X):

        self.u,self.s,self.vh = np.linalg.svd(X) #SVD decomposition

        self.total_var = (np.sum(self.s**2)) #Total variance in the dataset

        return

    #Plot number of components vs. variance explained
    def plot_pca(self):

        #Number of components
        x = np.arange(1,self.s.shape[0]+1)
        #Cumulative proportion of variance
        y = (np.cumsum(self.s**2))/self.total_var

        if (type(self.n) is float):
            #Get number of components needed for specified proportion
            self.n = np.min(np.where(y >= self.n)) + 1

        plt.plot(x, y, 'r-')
        plt.grid()
        plt.xlabel('Number of components')
        plt.ylabel('Variance explained')
        plt.title('Prinicpal Component Analysis - 2nd data matrix')
        #plt.savefig('PCA2.png')
        plt.show()


    #Project data on the principal component(s)
    def pca_transform(self, X):

        u1 = self.u[:,0:self.n]
        s1 = self.s[0:self.n]
        vh1 = self.vh[0:self.n,:]

        self.pc = vh1

        Y = X @ self.pc.T

        return Y
