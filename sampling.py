"""
==============================================================
Restricted Boltzmann Machine  with Parallel Tempering
==============================================================
Sample the visibile units (i.e. reconstructions) after trianing.
Set the visible units to data point and see mixing capabilities of different
sampling techniques.
"""
from __future__ import print_function
from rbm import RBM
from rbm import RBM_CD
from rbm import RBM_PT
from rbm import RBM_LPT
from rbm import RBM_LPTOC

print(__doc__)

# Authors: Yann N. Dauphin, Vlad Niculae, Gabriel Synnaeve
# License: BSD

import numpy as np
import matplotlib.pyplot as plt
import os.path
FIGS_DIR = 'figs'
DATA_DIR = 'data'

from scipy.ndimage import convolve
from sklearn import linear_model, datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_mldata
from sklearn.utils import shuffle

def savefig(fname, verbose=True):
    """
    Saves the current figure to file.
    """
    path = os.path.join('.', FIGS_DIR, fname)
    plt.savefig(path)
    if verbose:
        print("Figure saved as '{}'".format(path))

def loaddata(rbm_obj, rbm_name):
    rbm_obj.components_        = np.load(DATA_DIR + "/" + rbm_name + "_weights"      + dataset + '.npy')
    rbm_obj.intercept_visible_ = np.load(DATA_DIR + "/" + rbm_name + "_visible_bias" + dataset + '.npy')
    rbm_obj.intercept_hidden_  = np.load(DATA_DIR + "/" + rbm_name + "_hidden_bias"  + dataset + '.npy')
    #rbm_obj.log_like           = np.load(DATA_DIR + "/" + rbm_name + "_log_like"     + dataset + '.npy')



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
MNIST = True
if not MNIST:
    digits = datasets.load_digits()
    X = np.asarray(digits.data, 'float32')
    X, Y = nudge_dataset(X, digits.target)
else:
    mnist = fetch_mldata('MNIST original')
    X = mnist.data
    Y = mnist.target
    X,Y = shuffle(X,Y)

X = (X - np.min(X, 0)) / (np.max(X, 0) + 0.0001)  # 0-1 scaling
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    test_size=0.2,
                                                    random_state=0)


best_params = {'n_components':50, 'learning_rate':0.02, 'batch_size':100}
n_iter = 100
n_components = best_params['n_components']
learning_rate = best_params['learning_rate']
batch_size = best_params['batch_size']
verbose = True
random_state = 0
room_temp=0.8
n_temp=10

# Models we will use
rbm_pcd = RBM(random_state=random_state, verbose=verbose, learning_rate=learning_rate, n_iter=n_iter, n_components=n_components, batch_size=batch_size)
rbm_cd = RBM_CD(random_state=random_state, verbose=verbose, learning_rate=learning_rate, n_iter=n_iter, n_components=n_components, batch_size=batch_size, cd_k=1)
rbm_pt = RBM_PT(random_state=random_state, verbose=verbose, learning_rate=learning_rate, n_iter=n_iter, n_components=n_components, batch_size=batch_size,
    n_temperatures=n_temp, room_temp=room_temp)
rbm_lpt= RBM_LPT(random_state=random_state, verbose=verbose, learning_rate=learning_rate, n_iter=n_iter, n_components=n_components, batch_size=batch_size,
    n_temperatures=n_temp, room_temp=room_temp)
rbm_lptp = RBM_LPTOC(random_state=random_state, verbose=verbose, learning_rate=learning_rate, n_iter=n_iter, n_components=n_components, batch_size=batch_size,
    n_temperatures=n_temp, room_temp=room_temp)


# Loads data of the trained model
dataset = 'MNIST'
loaddata(rbm_pcd, "rbm_pcd")
loaddata(rbm_cd, "rbm_cd")
loaddata(rbm_pt, "rbm_pt")
loaddata(rbm_lpt, "rbm_lpt")
loaddata(rbm_lptp, "rbm_lptd")

i = np.random.randint(0, X.shape[0])
X0 = X[i]
print(i)

dim = 28
n = 10
n2 = n**2
ng = 50
# rbm_pcd.v_sample_ = X0
# plt.figure(figsize=(4.2, 4))
# plt.subplot(n, n, 1)
# plt.imshow(X0.reshape((dim, dim)), cmap=plt.cm.gray_r,interpolation='nearest')
# for i in range(1,n2):
#     plt.subplot(n, n, i + 1)
#     rbm_pcd.ngibbs(ng)
#     plt.imshow(rbm_pcd.expectation().reshape((dim, dim)), cmap=plt.cm.gray_r,
#                interpolation='nearest')
#     plt.xticks(())
#     plt.yticks(())
#     print("PCD: " + str(i) + " images generated.")
# plt.suptitle('100 samples extracted by RBM w/ PCD', fontsize=16)
# plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

print("\nDone sampling PCD\n")

rbm_pt.v_sample_ = np.repeat(X0.reshape(1,1,X_train[0].shape[0]), rbm_pt.n_temperatures, axis=0)
rbm_pt.temp = np.asarray([0.99**i for i in range(rbm_pt.n_temperatures)])
plt.figure(figsize=(4.2, 4))
plt.subplot(n, n, 1)
plt.imshow(X0.reshape((dim, dim)), cmap=plt.cm.gray_r,interpolation='nearest')
for i in range(1,n2):
    plt.subplot(n, n, i + 1)
    rbm_pt.ngibbs(ng)
    plt.imshow(rbm_pt.expectation().reshape((dim, dim)), cmap=plt.cm.gray_r,
               interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
    print("PT: " + str(i) + " images generated.")
plt.suptitle('100 samples extracted by RBM w/ PT', fontsize=16)
plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

print("\nDone sampling PT\n")

rbm_lpt.v_sample_ = np.repeat(X0.reshape(1,1,X_train[0].shape[0]), rbm_pt.n_temperatures, axis=0)
plt.figure(figsize=(4.2, 4))
plt.subplot(n, n, 1)
plt.imshow(X0.reshape((dim, dim)), cmap=plt.cm.gray_r,interpolation='nearest')
for i in range(1,n2):
    plt.subplot(n, n, i + 1)
    rbm_lpt.ngibbs(ng)
    plt.imshow(rbm_lpt.expectation().reshape((dim, dim)), cmap=plt.cm.gray_r,
               interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
    print("LPT: " + str(i) + " images generated.")
plt.suptitle('100 samples extracted by RBM w/ LPT', fontsize=16)
plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

print("\nDone sampling LPT\n")

rbm_lptp.v_sample_ = np.repeat(X0.reshape(1,1,X_train[0].shape[0]), rbm_pt.n_temperatures, axis=0)
plt.figure(figsize=(4.2, 4))
plt.subplot(n, n, 1)
plt.imshow(X0.reshape((dim, dim)), cmap=plt.cm.gray_r,interpolation='nearest')
for i in range(1,n2):
    plt.subplot(n, n, i + 1)
    rbm_lptp.ngibbs(ng)
    plt.imshow(rbm_lptp.expectation().reshape((dim, dim)), cmap=plt.cm.gray_r,
               interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
    print("LPTP: " + str(i) + " images generated.")
plt.suptitle('100 samples extracted by RBM w/ LPTP', fontsize=16)
plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)


print("\nDone sampling LPTP\n")
plt.show()
