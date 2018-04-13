# import the necessary packages
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.neural_network import BernoulliRBM
from sklearn.datasets import fetch_mldata
import numpy as np
import cPickle, gzip, numpy

mnist = fetch_mldata('MNIST original', data_home=custom_data_home)
print(mnist)

# Load the dataset
f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()
