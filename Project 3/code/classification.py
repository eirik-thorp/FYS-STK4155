import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn import svm
from sklearn import metrics
from functions import plot_data

import mlp
import pca

#Read in data
#data = pd.read_pickle('../data/preprocessed_data.pkl') #First data matrix
data = pd.read_pickle('../data/extended_preprocessed_data.pkl') #Second data matrix

features = list(data)[:-1]
X = data.loc[:, features]

#Incompatable (and irrelevant) data
X.drop(X.select_dtypes(['object']), inplace=True, axis=1)
X.drop(X.select_dtypes(['datetime64[ns]']), inplace=True, axis=1)

X = X.values
y = data.loc[:, ['label']].values

#Create labels
le = LabelEncoder()
le.fit(['Win', 'Draw', 'Defeat'])
y = le.transform(y.ravel())

#Split into training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80, test_size=0.20, random_state=1)

#Prinicpal component analysis
pca = pca.pca(0.95)

#Standardize the data
pca.scale_fit(X_train)
X_train = pca.scale_transform(X_train)
X_test = pca.scale_transform(X_test)

#Find the principal components
pca.pca_fit(X_train)
pca.plot_pca()
X_train = pca.pca_transform(X_train)
X_test = pca.pca_transform(X_test)


''' Logistic regression '''

C = np.array([0.01, 0.1, 0.5, 1, 2, 5])
test_scores = np.zeros(6)
train_scores = np.zeros(6)

for i, c in enumerate(C):
    logisticRegr = LogisticRegression(solver = 'liblinear', C=c)
    logisticRegr.fit(X_train, y_train)
    print('LogReg train score:', logisticRegr.score(X_train, y_train))
    print('LogReg test score:', logisticRegr.score(X_test, y_test))
    train_scores[i] = logisticRegr.score(X_train, y_train)
    test_scores[i] = logisticRegr.score(X_test, y_test)

plt.plot(C, train_scores, 'r-', label='train score')
plt.plot(C, test_scores, 'b-', label='test score')
plt.grid()
plt.legend()
plt.xlabel('Regularization parameter')
plt.ylabel('Accuracy')
plt.title('Accuracy of logreg on 1st data matrix')
plt.savefig('logreg1.png')
plt.show()

''' SVM with different kernels '''


C = np.array([0.01, 0.1, 0.5, 1, 2, 5])

test_scores_linear = np.zeros(6)
train_scores_linear = np.zeros(6)

test_scores_rbf = np.zeros(6)
train_scores_rbf = np.zeros(6)

test_scores_sigmoid = np.zeros(6)
train_scores_sigmoid = np.zeros(6)

test_scores_poly2 = np.zeros(6)
train_scores_poly2 = np.zeros(6)

test_scores_poly5 = np.zeros(6)
train_scores_poly5 = np.zeros(6)

for i, c in enumerate(C):

    clf = svm.SVC(kernel='linear', C=c)
    clf.fit(X_train, y_train)
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    train_scores_linear[i] = metrics.accuracy_score(y_train, y_pred_train)
    test_scores_linear[i] = metrics.accuracy_score(y_test, y_pred_test)

    clf = svm.SVC(kernel='rbf', C=c)
    clf.fit(X_train, y_train)
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    train_scores_rbf[i] = metrics.accuracy_score(y_train, y_pred_train)
    test_scores_rbf[i] = metrics.accuracy_score(y_test, y_pred_test)

    clf = svm.SVC(kernel='sigmoid', C=c)
    clf.fit(X_train, y_train)
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    train_scores_sigmoid[i] = metrics.accuracy_score(y_train, y_pred_train)
    test_scores_sigmoid[i] = metrics.accuracy_score(y_test, y_pred_test)

    clf = svm.SVC(kernel='poly', degree=2,C=c)
    clf.fit(X_train, y_train)
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    train_scores_poly2[i] = metrics.accuracy_score(y_train, y_pred_train)
    test_scores_poly2[i] = metrics.accuracy_score(y_test, y_pred_test)

    clf = svm.SVC(kernel='poly', degree=5,C=c)
    clf.fit(X_train, y_train)
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    train_scores_poly5[i] = metrics.accuracy_score(y_train, y_pred_train)
    test_scores_poly5[i] = metrics.accuracy_score(y_test, y_pred_test)

plt.plot(C, train_scores_linear, 'r-', label='linear train')
plt.plot(C, test_scores_linear, 'r--', label='linear test')

plt.plot(C, train_scores_rbf, 'b-', label='rbf train')
plt.plot(C, test_scores_rbf, 'b--', label='rbf test')

plt.plot(C, train_scores_sigmoid, 'g-', label='sigmoid train')
plt.plot(C, test_scores_sigmoid, 'g--', label='sigmoid test')

plt.plot(C, train_scores_poly2, 'y-', label='poly2 train')
plt.plot(C, test_scores_poly2, 'y--', label='poly2test')

plt.plot(C, train_scores_poly5, 'c-', label='poly5 train')
plt.plot(C, test_scores_poly5, 'c--', label='poly5 test')

plt.grid()
plt.legend()
plt.xlabel('Regularization parameter')
plt.ylabel('Accuracy')
plt.title('Accuracy of SVM with different kernels on 1st data matrix')
plt.savefig('SVM1.png')
plt.show()


''' Grid search using the multi-layer perceptron '''

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

nh = np.array([1, 2, 4, 8]) #hidden nodes
et = np.array([0.001, 0.01, 0.05, 0.1]) #Learning rate

train_accuracy = np.zeros((len(nh), len(et)))
test_accuracy = np.zeros((len(nh), len(et)))

for i, n in enumerate(nh):
    for j, e in enumerate(et):
        mlp1 = mlp.mlp(X_train, y_train, nhidden=n, eta=e, linear=False)
        mlp1.earlystopping(X_train, y_train, X_test, y_test)
        train_accuracy[i,j] = (mlp1.score(X_train, y_train))
        test_accuracy[i,j] = (mlp1.score(X_test, y_test))

plot_data(et, nh, train_accuracy)
plot_data(et, nh, test_accuracy)
