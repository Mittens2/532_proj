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
from rbm import RBM_LPTOC
import os.path

FIGS_DIR = 'figs'

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

def savefig(fname, verbose=True):
    path = os.path.join('.', FIGS_DIR, fname)
    plt.savefig(path)
    if verbose:
        print("Figure saved as '{}'".format(path))

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

# mnist = fetch_mldata('MNIST original')
# X = mnist.data
# Y = mnist.target
# X,Y = shuffle(X,Y)

X = (X - np.min(X, 0)) / (np.max(X, 0) + 0.0001)  # 0-1 scaling
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    test_size=0.2,
                                                random_state=0)

# Models we will use
best_params = {'n_iter':100, 'n_components':50, 'learning_rate':0.02}
n_iter = best_params['n_iter']
n_components = best_params['n_components']
learning_rate = best_params['learning_rate']
verbose = True
random_state = 0
rbm_pcd = RBM(random_state=random_state, verbose=verbose, learning_rate=learning_rate, n_iter=n_iter, n_components=n_components)
rbm_cd = RBM_CD(random_state=random_state, verbose=verbose, learning_rate=learning_rate, n_iter=n_iter, n_components=n_components, cd_k=1)
rbm_pt = RBM_PT(random_state=random_state, verbose=verbose, learning_rate=learning_rate, n_iter=n_iter, n_components=n_components,
    temp=np.array([0.8**i for i in range(6)]))
rbm_lpt= RBM_LPT(random_state=random_state, verbose=verbose, learning_rate=learning_rate, n_iter=n_iter, n_components=n_components,
    temp=np.array([0.8**i for i in range(6)]))
rbm_lptoc = RBM_LPTOC(random_state=random_state, verbose=verbose, learning_rate=learning_rate, n_iter=n_iter, n_components=n_components,
    temp=np.array([0.8**i for i in range(6)]))

# logistic1 = linear_model.LogisticRegression(C=6000.0)
# logistic2 = linear_model.LogisticRegression(C=6000.0)

#classifier1 = Pipeline(steps=[('rbm', rbm), ('logistic', logistic1)])
#classifier2 = Pipeline(steps=[('rbm', rbm_cd), ('logistic', logistic2)])
# #############################################################################
# Training
# rbm = RBM_CD()
# params = [{'n_iter': [100], 'n_components': [25, 50, 100],
#      'learning_rate': [0.001, 0.01, 0.1]}]
# rbm_cv = GridSearchCV(rbm, params)
# rbm_cv.fit(X_train, Y_train)
# best_params = rbm_cv.best_params_
# print("Best parameters set found on development set:")
# print()
# print(rbm_cv.best_params_)
# print()
# print("Grid scores on development set:")
# print()
# means = rbm_cv.cv_results_['mean_test_score']
# stds = rbm_cv.cv_results_['std_test_score']
# for mean, std, params in zip(means, stds, rbm_cv.cv_results_['params']):
#     print("%0.3f (+/-%0.03f) for %r"
#           % (mean, std * 2, params))

# Training RBM-Logistic Pipeline
#classifier1.fit(X_train, Y_train)
#
# rbm_pcd.fit(X_train, Y_train)
# np.save("data/rbm_pcd_weights",      rbm_pcd.components_)
# np.save("data/rbm_pcd_visible_bias", rbm_pcd.intercept_visible_)
# np.save("data/rbm_pcd_hidden_bias",  rbm_pcd.intercept_hidden_)
# plt.plot(np.arange(1, rbm_pcd.n_iter + 1), rbm_pcd.log_like, label='PCD')
#
# rbm_cd.fit(X_train, Y_train)
# np.save("data/rbm_cd_weights",      rbm_cd.components_)
# np.save("data/rbm_cd_visible_bias", rbm_cd.intercept_visible_)
# np.save("data/rbm_cd_hidden_bias",  rbm_cd.intercept_hidden_)
# plt.plot(np.arange(1, rbm_cd.n_iter + 1), rbm_cd.log_like, label='CD')

# rbm_pt.fit(X_train, Y_train)
# np.save("data/rbm_pt_weights",      rbm_pt.components_)
# np.save("data/rbm_pt_visible_bias", rbm_pt.intercept_visible_)
# np.save("data/rbm_pt_hidden_bias",  rbm_pt.intercept_hidden_)
# plt.plot(np.arange(1, rbm_pt.n_iter + 1), rbm_pt.log_like, label='PT')
#
rbm_lpt.fit(X_train, Y_train)
np.save("data/rbm_lpt_weights",      rbm_lpt.components_)
np.save("data/rbm_lpt_visible_bias", rbm_lpt.intercept_visible_)
np.save("data/rbm_lpt_hidden_bias",  rbm_lpt.intercept_hidden_)
plt.plot(np.arange(1, rbm_lpt.n_iter + 1), rbm_lpt.log_like, label='LPT')
#
# rbm_lptoc.fit(X_train, Y_train)
# np.save("data/rbm_pt_weights",      rbm_lptoc.components_)
# np.save("data/rbm_pt_visible_bias", rbm_lptoc.intercept_visible_)
# np.save("data/rbm_pt_hidden_bias",  rbm_lptoc.intercept_hidden_)
# plt.plot(np.arange(1, rbm_lptoc.n_iter + 1), rbm_lptoc.log_like, label='LPTP')

# plt.xlabel('iteration')
# plt.ylabel('log likelihood')
# title = 'TOY Log likelihood trend'
# plt.title(title)
# plt.legend()
# savefig(title + '.png')
# plt.show()


# rbm_pt.v_sample_= X[1,]
# plt.figure(figsize=(4.2, 4))
# rbm_pt.ngibbs(1000)
# e = rbm_pt.expectation()
# for i in range(5):
#     plt.subplot(1, 5, i + 1)
#     rbm_pt.ngibbs(100)
#     plt.imshow(e[i].reshape((28,28)), cmap=plt.cm.gray_r,
#                interpolation='nearest')
#     plt.xticks(())
#     plt.yticks(())
#
# plt.suptitle('100 components extracted by RBM w/ PT', fontsize=16)
# plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
#
# plt.show()

'''
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
'''
