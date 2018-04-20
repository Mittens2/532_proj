"""
==============================================================
Restricted Boltzmann Machine with Parallel Tempering
==============================================================
5 variants for RBM training
1) RBM with rbm_pcd
2) RBM with CD-k
3) RBM with PT
4) RBM with LPT
5) RBM with Deterministic Odd/Even (LPTD)
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
# digits = datasets.load_digits()
# X = np.asarray(digits.data, 'float32')
# X, Y = nudge_dataset(X, digits.target)

mnist = fetch_mldata('MNIST original')
X = mnist.data
Y = mnist.target
X,Y = shuffle(X,Y)

X = (X - np.min(X, 0)) / (np.max(X, 0) + 0.0001)  # 0-1 scaling
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    test_size=0.2,
                                                random_state=0)

logistic = linear_model.LogisticRegression(C=6000.0)
# #############################################################################
# Hyperparameter Tuning
#n_iter_train = 50
# rbm = RBM_CD(n_iter=n_iter_train)
# params = [{'n_components': [25, 50, 100],
#       'learning_rate': [0.001, 0.01, 0.1], 'batch_size': [1, 10, 100]}]
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

# Models we will use
best_params = {'n_components':50, 'learning_rate':0.02, 'batch_size':100}
n_iter = 50
n_components = best_params['n_components']
learning_rate = best_params['learning_rate']
batch_size = best_params['batch_size']
verbose = True
random_state = 0
room_temp=0.7
n_temp=6

rbm_pcd = RBM(random_state=random_state, verbose=verbose, learning_rate=learning_rate, n_iter=n_iter, n_components=n_components, batch_size=batch_size)
rbm_cd = RBM_CD(random_state=random_state, verbose=verbose, learning_rate=learning_rate, n_iter=n_iter, n_components=n_components, batch_size=batch_size, cd_k=1)
rbm_pt = RBM_PT(random_state=random_state, verbose=verbose, learning_rate=learning_rate, n_iter=n_iter, n_components=n_components, batch_size=batch_size,
    n_temperatures=n_temp, room_temp=room_temp)
rbm_lpt= RBM_LPT(random_state=random_state, verbose=verbose, learning_rate=learning_rate, n_iter=n_iter, n_components=n_components, batch_size=batch_size,
    n_temperatures=n_temp, room_temp=room_temp)
rbm_lptp = RBM_LPTOC(random_state=random_state, verbose=verbose, learning_rate=learning_rate, n_iter=n_iter, n_components=n_components, batch_size=batch_size,
    n_temperatures=n_temp, room_temp=room_temp)

# Training RBMs
dataset = 'MNIST'
rbm_pcd.fit(X_train, Y_train)
np.save("data/rbm_pcd_weights" + dataset, rbm_pcd.components_)
np.save("data/rbm_pcd_visible_bias" + dataset, rbm_pcd.intercept_visible_)
np.save("data/rbm_pcd_hidden_bias" + dataset, rbm_pcd.intercept_hidden_)
plt.plot(np.arange(1, rbm_pcd.n_iter + 1), rbm_pcd.log_like, label='PCD')

rbm_cd.fit(X_train, Y_train)
np.save("data/rbm_cd_weights" + dataset, rbm_cd.components_)
np.save("data/rbm_cd_visible_bias" + dataset, rbm_cd.intercept_visible_)
np.save("data/rbm_cd_hidden_bias" + dataset,  rbm_cd.intercept_hidden_)
plt.plot(np.arange(1, rbm_cd.n_iter + 1), rbm_cd.log_like, label='CD')

rbm_pt.fit(X_train, Y_train)
np.save("data/rbm_pt_weights" + dataset, rbm_pt.components_)
np.save("data/rbm_pt_visible_bias" + dataset, rbm_pt.intercept_visible_)
np.save("data/rbm_pt_hidden_bias" + dataset, rbm_pt.intercept_hidden_)
plt.plot(np.arange(1, rbm_pt.n_iter + 1), rbm_pt.log_like, label='PT')

rbm_lpt.fit(X_train, Y_train)
np.save("data/rbm_lpt_weights" + dataset, rbm_lpt.components_)
np.save("data/rbm_lpt_visible_bias" + dataset, rbm_lpt.intercept_visible_)
np.save("data/rbm_lpt_hidden_bias" + dataset, rbm_lpt.intercept_hidden_)
plt.plot(np.arange(1, rbm_lpt.n_iter + 1), rbm_lpt.log_like, label='LPT')

rbm_lptp.fit(X_train, Y_train)
np.save("data/rbm_lptd_weights" + dataset, rbm_lptp.components_)
np.save("data/rbm_lptd_visible_bias" + dataset, rbm_lptp.intercept_visible_)
np.save("data/rbm_lptd_hidden_bias" + dataset, rbm_lptp.intercept_hidden_)
plt.plot(np.arange(1, rbm_lptp.n_iter + 1), rbm_lptp.log_like, label='LPTD')

# Plot log likelihood
plt.xlabel('iteration')
plt.ylabel('log likelihood')
title = dataset + 'Log likelihood trend'
plt.title(title)
plt.legend()
savefig(title + '.png')
plt.show()
