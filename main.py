"""
==============================================================
Restricted Boltzmann Machine features for digit classification
==============================================================

For greyscale image data where pixel values can be interpreted as degrees of
blackness on a white background, like handwritten digit recognition, the
Bernoulli Restricted Boltzmann machine model (:class:`BernoulliRBM
<sklearn.neural_network.BernoulliRBM>`) can perform effective non-linear
feature extraction.

In order to learn good latent representations from a small dataset, we
artificially generate more labeled data by perturbing the training data with
linear shifts of 1 pixel in each direction.

This example shows how to build a classification pipeline with a BernoulliRBM
feature extractor and a :class:`LogisticRegression
<sklearn.linear_model.LogisticRegression>` classifier. The hyperparameters
of the entire model (learning rate, hidden layer size, regularization)
were optimized by grid search, but the search is not reproduced here because
of runtime constraints.

Logistic regression on raw pixel values is presented for comparison. The
example shows that the features extracted by the BernoulliRBM help improve the
classification accuracy.
"""
from __future__ import print_function
from rbm import RBM_CD
from rbm import RBM_PT

print(__doc__)

# Authors: Yann N. Dauphin, Vlad Niculae, Gabriel Synnaeve
# License: BSD

import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import convolve
from sklearn import linear_model, datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_mldata
from sklearn.utils import shuffle


# #############################################################################
# Setting up

def nudge_dataset(X, Y):
    """
    This produces a dataset 5 times bigger than the original one,
    by moving the 8x8 images in X around by 1px to left, right, down, up
    """
    direction_vectors = [
        [[0, 1, 0],
         [0, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [1, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 1],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 0],
         [0, 1, 0]]]

    shift = lambda x, w: convolve(x.reshape((8, 8)), mode='constant',
                                  weights=w).ravel()
    X = np.concatenate([X] +
                       [np.apply_along_axis(shift, 1, X, vector)
                        for vector in direction_vectors])
    Y = np.concatenate([Y for _ in range(5)], axis=0)
    return X, Y

# Load Data
digits = datasets.load_digits()
X = np.asarray(digits.data, 'float32')
X, Y = nudge_dataset(X, digits.target)

mnist = fetch_mldata('MNIST original')
X = mnist.data
Y = mnist.target
X,Y = shuffle(X,Y)

X = (X - np.min(X, 0)) / (np.max(X, 0) + 0.0001)  # 0-1 scaling

X = X[1:10000]
Y = Y[1:10000]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    test_size=0.2,
                                                    random_state=0)

# Models we will use
logistic1 = linear_model.LogisticRegression(C=6000.0)
logistic2 = linear_model.LogisticRegression(C=6000.0)
rbm = BernoulliRBM(random_state=0, verbose=True, learning_rate=0.02, n_iter=100, n_components=50)
rbm_cd = RBM_CD(random_state=0, verbose=True, learning_rate=0.05, n_iter=10, n_components=50, cd_k=7)
rbm_pt = RBM_PT(random_state=0, verbose=True, learning_rate=0.05, n_iter=10, n_components=50, temp=np.array([1-i/2 for i in range(2)]))

classifier1 = Pipeline(steps=[('rbm', rbm), ('logistic', logistic1)])
classifier2 = Pipeline(steps=[('rbm', rbm_cd), ('logistic', logistic2)])


# #############################################################################
# Training

# Just RBM
#rbm_cd.fit(X_train, Y_train)

# RBM with GridSearchCV
# parameters = [{'n_iter': [20], 'n_components': [100], 'learning_rate': [0.01, 0.05, 0.1], 'cd_k': [1, 3, 5]}]
# clf = GridSearchCV(rbm_cd, parameters)
# clf.fit(X_train, Y_train)
# print(sorted(clf.cv_results_.keys()))

# Training RBM-Logistic Pipeline
#classifier1.fit(X_train, Y_train)
#rbm_cd.fit(X_train, Y_train)
rbm_pt.fit(X_train, Y_train)

v = X[1,]
plt.figure(figsize=(4.2, 4))
for i, comp in enumerate(rbm_cd.components_):
    plt.subplot(10, 10, i + 1)
    plt.imshow(rbm_cd.continuous_gibbs(v).reshape((28, 28)), cmap=plt.cm.gray_r,
               interpolation='nearest')
    v = rbm_cd.gibbs(v)
    plt.xticks(())
    plt.yticks(())
plt.suptitle('100 components extracted by RBM w/ PCD', fontsize=16)
plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

v = X[1,]
plt.figure(figsize=(4.2, 4))
for i, comp in enumerate(rbm_cd.components_):
    plt.subplot(10, 10, i + 1)
    plt.imshow(rbm_cd.continuous_gibbs(v).reshape((28, 28)), cmap=plt.cm.gray_r,
               interpolation='nearest')
    v = rbm_cd.ngibbs(v, 50)
    plt.xticks(())
    plt.yticks(())
plt.suptitle('100 components extracted by RBM w/ PCD', fontsize=16)
plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)


#plt.imshow(rbm.gibbs(X[1,]).reshape(8,8))
plt.show()


# #############################################################################
# Evaluation

print()
print("Logistic regression using PCD RBM features:\n%s\n" % (
    metrics.classification_report(
        Y_test,
        classifier1.predict(X_test))))

print("Logistic regression using CD RBM features:\n%s\n" % (
    metrics.classification_report(
        Y_test,
        classifier2.predict(X_test))))
#
# print("Logistic regression using raw pixel features:\n%s\n" % (
#     metrics.classification_report(
#         Y_test,
#         logistic_classifier.predict(X_test))))

# #############################################################################
# Plotting

plt.figure(figsize=(4.2, 4))
for i, comp in enumerate(rbm_cd.components_):
    plt.subplot(10, 10, i + 1)
    plt.imshow(comp.reshape((8, 8)), cmap=plt.cm.gray_r,
               interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
plt.suptitle('100 components extracted by RBM w/ CD', fontsize=16)
plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

plt.figure(figsize=(4.2, 4))
for i, comp in enumerate(rbm.components_):
    plt.subplot(10, 10, i + 1)
    plt.imshow(comp.reshape((8, 8)), cmap=plt.cm.gray_r,
               interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
plt.suptitle('100 components extracted by RBM w/ PCD', fontsize=16)
plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

plt.show()
