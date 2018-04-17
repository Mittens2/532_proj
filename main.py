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
from rbm import RBM
from rbm import RBM_CD
from rbm import RBM_PT
from rbm import RBM_LPT

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
rbm_pcd = RBM    (random_state=0, verbose=True, learning_rate=0.02, batch_size=10, n_iter=100, n_components=100)
rbm_cd  = RBM_CD (random_state=0, verbose=True, learning_rate=0.02, batch_size=10, n_iter=100, n_components=100, cd_k=1)
rbm_pt  = RBM_PT (random_state=0, verbose=True, learning_rate=0.02, batch_size=10, n_iter=100, n_components=100, temp=np.array([0.9**i for i in range(10)]))
rbm_lpt = RBM_LPT(random_state=0, verbose=True, learning_rate=0.02, batch_size=10, n_iter=100, n_components=100, temp=np.array([0.9**i for i in range(10)]))

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

rbm_pcd.fit(X_train, Y_train)
np.save("data/rbm_pcd_weights",      rbm.components_)
np.save("data/rbm_pcd_visible_bias", rbm.intercept_visible_)
np.save("data/rbm_pcd_hidden_bias",  rbm.intercept_hidden_)

rbm_cd.fit(X_train, Y_train)
np.save("data/rbm_cd_weights",      rbm_cd.components_)
np.save("data/rbm_cd_visible_bias", rbm_cd.intercept_visible_)
np.save("data/rbm_cd_hidden_bias",  rbm_cd.intercept_hidden_)

rbm_pt.fit(X_train, Y_train)
np.save("data/rbm_pt_weights",      rbm_pt.components_)
np.save("data/rbm_pt_visible_bias", rbm_pt.intercept_visible_)
np.save("data/rbm_pt_hidden_bias",  rbm_pt.intercept_hidden_)

rbm_lpt.fit(X_train, Y_train)
np.save("data/rbm_lpt_weights",      rbm_lpt.components_)
np.save("data/rbm_lpt_visible_bias", rbm_lpt.intercept_visible_)
np.save("data/rbm_lpt_hidden_bias",  rbm_lpt.intercept_hidden_)

'''
rbm_pt.v_sample_ = np.repeat(X[0].reshape(1,1,X[0].shape[0]), rbm_pt.n_temperatures, axis=0)
plt.figure(figsize=(4.2, 4))
rbm_pt.ngibbs(1000)
e = rbm_pt.expectation()
for i in range(5):
    plt.subplot(1, 5, i + 1)
    rbm_pt.ngibbs(100)
    plt.imshow(e[i].reshape((28,28)), cmap=plt.cm.gray_r,
               interpolation='nearest')
    plt.xticks(())
    plt.yticks(())

plt.suptitle('100 components extracted by RBM w/ PT', fontsize=16)
plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

plt.show()
'''






rbm_pcd.v_sample_ = X_train[0]
plt.figure(figsize=(4.2, 4))
plt.subplot(5, 5, 1)
plt.imshow(X_train[0].reshape((28, 28)), cmap=plt.cm.gray_r,interpolation='nearest')
for i in range(1,25):
    plt.subplot(5, 5, i + 1)
    rbm_pcd.ngibbs(100)
    plt.imshow(rbm_pcd.expectation().reshape((28, 28)), cmap=plt.cm.gray_r,
               interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
plt.suptitle('100 components extracted by RBM w/ PCD', fontsize=16)
plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

rbm_cd.v_sample_ = X_train[0]
plt.figure(figsize=(4.2, 4))
plt.subplot(5, 5, 1)
plt.imshow(X_train[0].reshape((28, 28)), cmap=plt.cm.gray_r,interpolation='nearest')
for i in range(1,25):
    plt.subplot(5, 5, i + 1)
    rbm_cd.ngibbs(100)
    plt.imshow(rbm_cd.expectation().reshape((28, 28)), cmap=plt.cm.gray_r,
               interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
plt.suptitle('100 components extracted by RBM w/ CD', fontsize=16)
plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

rbm_pt.v_sample_ = np.repeat(X[0].reshape(1,1,X[0].shape[0]), rbm_pt.n_temperatures, axis=0)
plt.figure(figsize=(4.2, 4))
plt.subplot(5, 5, 1)
plt.imshow(X_train[0].reshape((28, 28)), cmap=plt.cm.gray_r,interpolation='nearest')
for i in range(1,25):
    plt.subplot(5, 5, i + 1)
    rbm_pt.ngibbs(100)
    plt.imshow(rbm_pt.expectation().reshape((28, 28)), cmap=plt.cm.gray_r,
               interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
plt.suptitle('100 components extracted by RBM w/ PT', fontsize=16)
plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

rbm_lpt.v_sample_ = np.repeat(X[0].reshape(1,1,X[0].shape[0]), rbm_pt.n_temperatures, axis=0)
plt.figure(figsize=(4.2, 4))
plt.subplot(5, 5, 1)
plt.imshow(X_train[0].reshape((28, 28)), cmap=plt.cm.gray_r,interpolation='nearest')
for i in range(1,25):
    plt.subplot(5, 5, i + 1)
    rbm_lpt.ngibbs(100)
    plt.imshow(rbm_lpt.expectation().reshape((28, 28)), cmap=plt.cm.gray_r,
               interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
plt.suptitle('100 components extracted by RBM w/ LPT', fontsize=16)
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
