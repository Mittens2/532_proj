from sklearn.neural_network import BernoulliRBM
from sklearn.utils.extmath import safe_sparse_dot
from scipy.special import expit  # logistic function
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils import check_array
from sklearn.utils import check_random_state
from sklearn.utils import gen_even_slices
from sklearn.utils import issparse
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.extmath import log_logistic
from sklearn.utils.validation import check_is_fitted
import numpy as np
import time
from sklearn.externals.six.moves import xrange

def exchange_state(RBM1, RBM2):
    exchange = np.exp((1 / RBM1.temp - 1/ RBM2.temp) * (RBM1._free_energy))

# RBM with classic CD as opposed to PCD
class RBM_CD(BernoulliRBM):
    def __init__(self, n_components=256, learning_rate=0.1, batch_size=10,
                 n_iter=10, verbose=0, random_state=None, cd_k=1):
        self.n_components = n_components
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.verbose = verbose
        self.random_state = random_state
        self.cd_k = cd_k

    def _fit(self, v_pos, rng, ):
        """Inner fit for one mini-batch.
        Adjust the parameters to maximize the likelihood of v using
        Stochastic Maximum Likelihood (SML).
        Parameters
        ----------
        v_pos : array-like, shape (n_samples, n_features)
            The data to use for training.
        rng : RandomState
            Random number generator to use for sampling.
        """
        h_pos = self._mean_hiddens(v_pos)
        h_neg = h_pos
        for i in range(self.cd_k):
            h_neg[rng.uniform(size=h_neg.shape) < h_neg] = 1.0  # sample binomial
            self.h_samples_ = np.floor(h_neg, h_neg)
            v_neg = self._sample_visibles(self.h_samples_, rng)
            h_neg = self._mean_hiddens(v_neg)

        lr = float(self.learning_rate) / v_pos.shape[0]
        update = safe_sparse_dot(v_pos.T, h_pos, dense_output=True).T
        update -= np.dot(h_neg.T, v_neg)
        self.components_ += lr * update
        self.intercept_hidden_ += lr * (h_pos.sum(axis=0) - h_neg.sum(axis=0))
        self.intercept_visible_ += lr * (np.asarray(
            v_pos.sum(axis=0)).squeeze() -
                                         v_neg.sum(axis=0))

        h_neg[rng.uniform(size=h_neg.shape) < h_neg] = 1.0  # sample binomial
        self.h_samples_ = np.floor(h_neg, h_neg)

    def score(self, X, y):
        return self.score_samples(X).mean()

    def _mean_visibles(self, h):
        """Sample from the distribution P(v|h).
        Parameters
        ----------
        h : array-like, shape (n_samples, n_components)
            Values of the hidden layer to sample from.
        rng : RandomState
            Random number generator to use.
        Returns
        -------
        v : array-like, shape (n_samples, n_features)
            Values of the visible layer.
        """
        p = np.dot(h, self.components_)
        p += self.intercept_visible_
        expit(p, out=p)
        return (p)

    def continuous_gibbs(self, v):
        """Perform one Gibbs sampling step.
        Parameters
        ----------
        v : array-like, shape (n_samples, n_features)
            Values of the visible layer to start from.
        Returns
        -------
        v_new : array-like, shape (n_samples, n_features)
            Values of the visible layer after one Gibbs step.
        """
        if not hasattr(self, "random_state_"):
            self.random_state_ = check_random_state(self.random_state)
        h_ = self._sample_hiddens(v, self.random_state_)
        v_ = self._mean_visibles(h_)

        return v_

    def ngibbs(self, v, n):
        for i in range(n):
            v = self.gibbs(v)
        return v

# RBM for parallel tempering
class RBM_PT(BernoulliRBM):
    def __init__(self, n_components=256, learning_rate=0.1, batch_size=10,
                 n_iter=10, verbose=0, random_state=None, temp=np.array([1-i/5 for i in range(5)])):
        self.n_components = n_components
        self.temp = temp
        self.n_temperatures = len(temp)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.verbose = verbose
        self.random_state = random_state

    def _free_energy(self, v):
        """Computes the free energy F(v) = - log sum_h exp(-E(v,h)).
        Parameters
        ----------
        v : array-like, shape (n_temperatures, n_samples, n_features)
            Values of the visible layer.
        Returns
        -------
        free_energy : array-like, shape (n_temperatures, n_samples,)
            The value of the free energy.
        """
        ##print(v.shape)
        #print(self.intercept_visible_[:,:,None].shape)
        #print(self.temp[:,None,None].shape)

        return (- (v @ self.intercept_visible_[:,:,None]) * self.temp[:,None,None]
            - np.logaddexp(0, (v @ np.transpose(self.components_, (0,2,1))
                           + self.intercept_hidden_[:,None,:]) * self.temp[:,None,None]).sum(axis=1))


    def fit(self, X, y=None):
        """Fit the model to the data X.
        Parameters
        ----------
        X : {array-like, sparse matrix} shape (n_samples, n_features)
            Training data.
        Returns
        -------
        self : BernoulliRBM
            The fitted model.
        """
        X = check_array(X, accept_sparse='csr', dtype=np.float64)
        n_samples = X.shape[0]
        rng = check_random_state(self.random_state)

        self.components_ = np.asarray(
            rng.normal(0, 0.01, (self.n_temperatures, self.n_components, X.shape[1])),
            order='F')
        self.intercept_hidden_ = np.zeros((self.n_temperatures, self.n_components))
        self.intercept_visible_ = np.zeros((self.n_temperatures, X.shape[1]))
        self.h_samples_ = np.zeros((self.n_temperatures, self.batch_size, self.n_components))

        n_batches = int(np.ceil(float(n_samples) / self.batch_size))
        batch_slices = list(gen_even_slices(n_batches * self.batch_size,
                                            n_batches, n_samples))
        verbose = self.verbose
        begin = time.time()
        for iteration in xrange(1, self.n_iter + 1):
            for batch_slice in batch_slices:
                v = np.repeat(X[batch_slice].reshape(1,X[batch_slice].shape[0], X[batch_slice].shape[1]), self.n_temperatures, axis=0)
                self._fit(v, rng)

            if verbose:
                end = time.time()
                print("[%s] Iteration %d, pseudo-likelihood = %.2f,"
                      " time = %.2fs"
                      % (type(self).__name__, iteration,
                         self.score_samples(X).mean(),
                         end - begin))
                begin = end

        return self

    def _fit(self, v_pos, rng, ):
        """Inner fit for one mini-batch.
        Adjust the parameters to maximize the likelihood of v using
        Stochastic Maximum Likelihood (SML).
        Parameters
        ----------
        v_pos : array-like, shape (n_samples, n_features, n_temperatures)
            The data to use for training.
        rng : RandomState
            Random number generator to use for sampling.
        """
        h_pos = self._mean_hiddens(v_pos)
        v_neg = self._sample_visibles(self.h_samples_, rng)
        h_neg = self._mean_hiddens(v_neg)

        lr = float(self.learning_rate) / v_pos.shape[0]
        update = np.transpose(np.transpose(v_pos, (0,2,1)) @ h_pos, (0,2,1))
        update -= np.transpose(h_neg, (0,2,1)) @ v_neg
        self.components_ += lr * update
        self.intercept_hidden_ += lr * (h_pos.sum(axis=1) - h_neg.sum(axis=1))
        self.intercept_visible_ += lr * (np.asarray(
                                         v_pos.sum(axis=1)).squeeze() -
                                         v_neg.sum(axis=1))

        h_neg[rng.uniform(size=h_neg.shape) < h_neg] = 1.0  # sample binomial
        self.h_samples_ = np.floor(h_neg, h_neg)

    def _mean_hiddens(self, v):
        """Computes the probabilities P(h=1|v).
        Parameters
        ----------
        v : array-like, shape (n_temperatures, n_samples, n_features)
            Values of the visible layer.
        Returns
        -------
        h : array-like, shape (n_temperatures, n_samples, n_components)
            Corresponding mean field values for the hidden layer.
        """
        p = v @ np.transpose(self.components_, (0,2,1))
        p += self.intercept_hidden_[:, None, :]
        p *= self.temp[:,None,None]
        return expit(p, out=p)

    def _sample_visibles(self, h, rng):
        """Sample from the distribution P(v|h).
        Parameters
        ----------
        h : array-like, shape (n_temperatures, n_samples, n_components)
            Values of the hidden layer to sample from.
        rng : RandomState
            Random number generator to use.
        Returns
        -------
        v : array-like, shape (n_temperatures, n_samples, n_features)
            Values of the visible layer.
        """
        p = h @ self.components_
        p += self.intercept_visible_[:, None, :]
        p *= self.temp[:,None,None]
        expit(p, out=p)
        return (rng.random_sample(size=p.shape) < p)



    def score_samples(self, X):
        """Compute the pseudo-likelihood of X.
        Parameters
        ----------
        X : {array-like, sparse matrix} shape (n_samples, n_features)
            Values of the visible layer. Must be all-boolean (not checked).
        Returns
        -------
        pseudo_likelihood : array-like, shape (n_samples,)
            Value of the pseudo-likelihood (proxy for likelihood).
        Notes
        -----
        This method is not deterministic: it computes a quantity called the
        free energy on X, then on a randomly corrupted version of X, and
        returns the log of the logistic function of the difference.
        """
        check_is_fitted(self, "components_")

        v = check_array(X, accept_sparse='csr')
        rng = check_random_state(self.random_state)
        print(v.shape)

        # Randomly corrupt one feature in each sample in v.
        ind = (np.arange(v.shape[0]),
               rng.randint(0, v.shape[1], v.shape[0]))
        if issparse(v):
            data = -2 * v[ind] + 1
            v_ = v + sp.csr_matrix((data.A.ravel(), ind), shape=v.shape)
        else:
            v_ = v.copy()
            v_[ind] = 1 - v_[ind]

        fe = self._free_energy(v)
        fe_ = self._free_energy(v_)
        return v.shape[1] * log_logistic(fe_ - fe)
