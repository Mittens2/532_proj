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

class RBM(BernoulliRBM):

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
            rng.normal(0, 0.01, (self.n_components, X.shape[1])),
            order='F')
        self.intercept_hidden_ = np.zeros(self.n_components, )
        self.intercept_visible_ = np.zeros(X.shape[1], )
        self.h_samples_ = np.zeros((self.batch_size, self.n_components))

        n_batches = int(np.ceil(float(n_samples) / self.batch_size))
        batch_slices = list(gen_even_slices(n_batches * self.batch_size,
                                            n_batches, n_samples))
        verbose = self.verbose
        begin = time.time()
        log_like = np.zeros(self.n_iter)
        for iteration in xrange(1, self.n_iter + 1):
            for batch_slice in batch_slices:
                self._fit(X[batch_slice], rng)

            if verbose:
                end = time.time()
                log_like[iteration - 1] = self.score_samples(X).mean()
                print("[%s] Iteration %d, pseudo-likelihood = %.2f,"
                      " time = %.2fs"
                      % (type(self).__name__, iteration,
                         self.score_samples(X).mean(), end - begin))
                begin = end

        self.log_like = log_like

        return self

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

    def expectation(self):
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
        h_ = self._sample_hiddens(self.v_sample_, self.random_state_)
        v_ = self._mean_visibles(h_)
        return v_

    def ngibbs(self, n):
        if not hasattr(self, "random_state_"):
            self.random_state_ = check_random_state(self.random_state)
        v_ = self.v_sample_
        for i in range(n):
            h_ = self._sample_hiddens(v_, self.random_state_)
            v_ = self._sample_visibles(h_, self.random_state_)
        self.v_sample_ = v_

        return v_


# RBM with classic CD as opposed to PCD
class RBM_CD(RBM):
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
            rng.normal(0, 0.01, (self.n_components, X.shape[1])),
            order='F')
        self.intercept_hidden_ = np.zeros(self.n_components, )
        self.intercept_visible_ = np.zeros(X.shape[1], )
        self.h_samples_ = np.zeros((self.batch_size, self.n_components))

        n_batches = int(np.ceil(float(n_samples) / self.batch_size))
        batch_slices = list(gen_even_slices(n_batches * self.batch_size,
                                            n_batches, n_samples))
        verbose = self.verbose
        begin = time.time()
        log_like = np.zeros(self.n_iter)
        for iteration in xrange(1, self.n_iter + 1):
            for batch_slice in batch_slices:
                self._fit(X[batch_slice], rng)

            if verbose:
                end = time.time()
                log_like[iteration - 1] = self.score_samples(X).mean()
                print("[%s] Iteration %d, pseudo-likelihood = %.2f,"
                      " time = %.2fs"
                      % (type(self).__name__, iteration,
                         self.score_samples(X).mean(), end - begin))
                begin = end

        self.log_like = log_like

        return self

# RBM for parallel tempering
class RBM_PT(BernoulliRBM):
    def __init__(self, n_components=256, learning_rate=0.1, batch_size=10,
                 n_iter=10, verbose=0, random_state=None, temp=np.array([1-i/5 for i in range(5)])):
        self.n_components = n_components
        self.temp = temp
        self.n_temperatures = temp.shape[0]
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.verbose = verbose
        self.random_state = random_state
        self.ex_ind = self.n_temperatures - 1

    def _free_energy(self, v, i):
        """Computes the free energy F(v) = - log sum_h exp(-E(v,h)).
        Parameters
        ----------
        v : array-like, shape (n_samples, n_features)
            Values of the visible layer.
        Returns
        -------
        free_energy : array-like, shape (n_samples,)
            The value of the free energy.
        """
        return (- safe_sparse_dot(v, self.intercept_visible_)
        - np.logaddexp(0, safe_sparse_dot(v, self.components_.T)
                       + self.intercept_hidden_).sum(axis=1))


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
            rng.normal(0, 0.01, (self.n_components, X.shape[1])),
            order='F')
        self.intercept_hidden_ = np.zeros(self.n_components)
        self.intercept_visible_ = np.zeros(X.shape[1])
        self.h_samples_ = np.zeros((self.n_temperatures, self.batch_size, self.n_components))

        n_batches = int(np.ceil(float(n_samples) / self.batch_size))
        batch_slices = list(gen_even_slices(n_batches * self.batch_size,
                                            n_batches, n_samples))
        verbose = self.verbose
        begin = time.time()
        log_like = np.zeros(self.n_iter)
        for iteration in xrange(1, self.n_iter + 1):
            for batch_slice in batch_slices:
                v = np.repeat(X[batch_slice].reshape(1,X[batch_slice].shape[0], X[batch_slice].shape[1]), self.n_temperatures, axis=0)
                self._fit(v, rng)

            if verbose:
                end = time.time()
                log_like[iteration - 1] = self.score_samples(X).mean()
                print("[%s] Iteration %d, pseudo-likelihood = %.2f,"
                      " time = %.2fs"
                      % (type(self).__name__, iteration,
                         self.score_samples(X).mean(),
                         end - begin))
                begin = end

        self.log_like = log_like
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
        self.exchange(v_neg, h_neg, rng)

        update = (v_pos[0].T @ h_pos[0]).T
        update -= h_neg[0].T @ v_neg[0]
        self.components_ += lr * update
        self.intercept_hidden_ += lr * (h_pos[0].sum(axis=0) - h_neg[0].sum(axis=0))
        self.intercept_visible_ += lr * (np.asarray(
                                         v_pos[0].sum(axis=0)).squeeze() -
                                         v_neg[0].sum(axis=0))

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
        p = v @ self.components_.T
        p += self.intercept_hidden_[None, None, :]
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
        p += self.intercept_visible_[None, None, :]
        p *= self.temp[:,None,None]
        expit(p, out=p)
        return (rng.random_sample(size=p.shape) < p)

    def _mean_visibles(self, h):
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
        p += self.intercept_visible_[None, None, :]
        p *= self.temp[:,None,None]
        expit(p, out=p)
        return p

    def _sample_hiddens(self, v, rng):
        """Sample from the distribution P(h|v).
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
        p = v @ self.components_.T
        p += self.intercept_hidden_[None, None, :]
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
        pseudo_likelihood : array-like, shape (n_temperatures, n_samples,)
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

        # Randomly corrupt one feature in each sample in v.
        ind = (np.arange(v.shape[0]),
               rng.randint(0, v.shape[1], v.shape[0]))
        if issparse(v):
            data = -2 * v[ind] + 1
            v_ = v + sp.csr_matrix((data.A.ravel(), ind), shape=v.shape)
        else:
            v_ = v.copy()
            v_[ind] = 1 - v_[ind]

        fe = self._free_energy(v, 0)
        fe_ = self._free_energy(v_, 0)
        return v.shape[1] * log_logistic(fe_ - fe)

    def expectation(self):
        if not hasattr(self, "random_state_"):
            self.random_state_ = check_random_state(self.random_state)
        h_ = self._sample_hiddens(self.v_sample_, self.random_state_)
        v_ = self._mean_visibles(h_)
        return v_[0]

    def ngibbs(self, n):
        if not hasattr(self, "random_state_"):
            self.random_state_ = check_random_state(self.random_state)
        v_ = self.v_sample_
        for i in range(n):
            h_ = self._sample_hiddens(v_, self.random_state_)
            v_ = self._sample_visibles(h_, self.random_state_)
            self.exchange(v_, h_, self.random_state_)
        self.v_sample_ = v_

        return v_[0]

    def exchange(self, v, h, rng):
        """
        Propose an exachage between two different parallel chains.
        Have a lifted parameter ex_dir which saves the direction of the swap.
        Accept with probability (1/T_i - 1/T_j) * (E_i(v_i, h_i) - E(v_j, h_j))
        Parameters
        ----------
        v : array-like, shape (n_temperatures, n_samples, n_features)
            Values of the visible layer.
        h : array-like, shape (n_temperatures, n_samples, n_components)
            Values of the hidden layer.
        rng : RandomState
            Random number generator to use.
        """
        i = self.ex_ind
        if i == self.n_temperatures - 1:
            j = i - 1
        elif i == 0:
            j = 1
        else:
            j = i + np.random.randint(0, 2) * 2 - 1
        en1 = -(v[i] @ self.intercept_visible_ + h[i] @ self.intercept_hidden_ \
            + np.sum((h[i] @ self.components_) * v[i], axis=1))
        en2 = -(v[j] @ self.intercept_visible_ + h[j] @ self.intercept_hidden_ \
            + np.sum(h[j] @ self.components_ * v[j], axis=1))
        # prob = (self.temp[i] - self.temp[j]) * (np.mean(en1) - np.mean(en2))
        # rand = np.log(rng.uniform())
        # if prob > rand:
        #     v_copy = np.copy(v[i])
        #     v[i] = v[j]
        #     v[j] = v_copy
        #     h_copy = np.copy(h[i])
        #     h[i] = h[j]
        #     h[j] = h_copy
        prob = (self.temp[i] - self.temp[j]) * (en1 - en2)
        rand = np.log(rng.uniform(prob.shape[0]))
        v_copy = v[i][prob > rand]
        v[i][prob > rand] = v[j][prob > rand]
        v[j][prob > rand] = v_copy
        h_copy = h[i][prob > rand]
        h[i][prob > rand] = h[j][prob > rand]
        h[j][prob > rand] = h_copy
        self.ex_ind = j


class RBM_LPT(RBM_PT):

    def __init__(self, n_components=256, learning_rate=0.1, batch_size=10,
                 n_iter=10, verbose=0, random_state=None, temp=np.array([1-i/5 for i in range(5)])):
        self.n_components = n_components
        self.temp = temp
        self.n_temperatures = temp.shape[0]
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.verbose = verbose
        self.random_state = random_state
        # self.ex_ind = self.n_temperatures - 1
        # self.ex_dir = -1
        self.ex_ind = (np.ones(self.batch_size) * self.n_temperatures - 1).astype(int)
        self.ex_dir = - np.ones(self.batch_size).astype(int)


    def exchange(self, v, h, rng):
        """
        Propose an exachage between two different parallel chains.
        Have a lifted parameter ex_dir which saves the direction of the swap.
        Accept with probability (1/T_i - 1/T_j) * (E_i(v_i, h_i) - E(v_j, h_j))
        Parameters
        ----------
        v : array-like, shape (n_temperatures, n_samples, n_features)
            Values of the visible layer.
        h : array-like, shape (n_temperatures, n_samples, n_components)
            Values of the hidden layer.
        rng : RandomState
            Random number generator to use.
        """
        # i = self.ex_ind
        # if i == self.n_temperatures - 1:
        #     j = i - 1
        #     self.ex_dir = -1
        # elif i == 0:
        #     j = 1
        #     self.ex_dir = 1
        # else:
        #     j = i + self.ex_dir
        # en1 = -(v[i] @ self.intercept_visible_ + h[i] @ self.intercept_hidden_ \
        #     + np.sum((h[i] @ self.components_) * v[i], axis=1))
        # en2 = -(v[j] @ self.intercept_visible_ + h[j] @ self.intercept_hidden_ \
        #     + np.sum(h[j] @ self.components_ * v[j], axis=1))
        # prob = (self.temp[i] - self.temp[j]) * (np.mean(en1) - np.mean(en2))
        # rand = np.log(rng.uniform())
        # if prob > rand:
        #     v_copy = np.copy(v[i])
        #     v[i] = v[j]
        #     v[j] = v_copy
        #     h_copy = np.copy(h[i])
        #     h[i] = h[j]
        #     h[j] = h_copy
        # else:
        #     self.ex_dir = -self.ex_dir

        lift = self.ex_dir
        i = self.ex_ind
        j = np.zeros(self.batch_size).astype(int)

        # edge cases
        j[i == self.n_temperatures - 1] = self.n_temperatures - 2
        lift[i == self.n_temperatures - 1] = - 1
        j[i == 0] = 1
        lift[i == 0] = 1
        j[j == 0] = i[j == 0] + lift[j == 0]

        ind_i = [tuple(i), tuple(range(10))]
        ind_j = [tuple(j), tuple(range(10))]
        en1 = -(v[ind_i] @ self.intercept_visible_ + h[ind_i] @ self.intercept_hidden_ \
            + np.sum(h[ind_i] @ self.components_ * v[ind_i], axis=1))
        en2 = -(v[ind_j] @ self.intercept_visible_ + h[ind_j] @ self.intercept_hidden_ \
            + np.sum(h[ind_j] @ self.components_ * v[ind_j], axis=1))

        prob = (self.temp[i] - self.temp[j]) * (en1 - en2)
        rand = np.log(rng.uniform(prob.shape[0]))
        v_copy = v[ind_i][prob > rand]
        v[ind_i][prob > rand] = v[ind_j][prob > rand]
        v[ind_j][prob > rand] = v_copy
        h_copy = h[ind_i][prob > rand]
        h[ind_i][prob > rand] = h[ind_j][prob > rand]
        h[ind_j][prob > rand] = h_copy
        self.ex_dir[prob > rand] *= -1
        self.ex_ind = j

# RBM with LPT on crack
class RBM_LPTOC(RBM_PT):
    def __init__(self, n_components=256, learning_rate=0.1, batch_size=10,
                 n_iter=10, verbose=0, random_state=None, temp=np.array([1-i/5 for i in range(5)])):
        self.n_components = n_components
        self.temp = temp
        self.n_temperatures = temp.shape[0]
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.verbose = verbose
        self.random_state = random_state
        self.ex_iter = 0

    def exchange(self, v, h, rng):
        """
        Propose an exchange all adjacent parallel chains.

        Accept with probability (1/T_i - 1/T_j) * (E_i(v_i, h_i) - E(v_j, h_j))
        Parameters
        ----------
        v : array-like, shape (n_temperatures, n_samples, n_features)
            Values of the visible layer.
        h : array-like, shape (n_temperatures, n_samples, n_components)
            Values of the hidden layer.
        rng : RandomState
            Random number generator to use.
        """
        i = self.ex_iter % 2
        # Get all energies
        en = -(v @ self.intercept_visible_ + h @ self.intercept_hidden_ \
            + np.sum(h @ self.components_ * v, axis=2))
        # prob = (self.temp[i:-1:2] - self.temp[i+1::2]) * (np.mean(en[i:-1:2], axis=1) - np.mean(en[i+1::2], axis=1))
        # rand = np.log(rng.uniform(size = prob.shape[0]))
        # v_copy = np.copy(v[i:-1:2][prob > rand])
        # v[i:-1:2][prob > rand] = v[i+1::2][prob > rand]
        # v[i+1::2][prob > rand] = v_copy
        # h_copy = np.copy(h[i:-1:2][prob > rand])
        # h[i:-1:2][prob > rand] = h[i+1::2][prob > rand]
        # h[i+1::2][prob > rand] = h_copy
        prob = np.multiply((self.temp[i:-1:2] - self.temp[i+1::2]), (en[i:-1:2] - en[i+1::2]).T).T
        rand = np.log(rng.uniform(size = prob.shape))
        v_copy = np.copy(v[i:-1:2][prob > rand])
        v[i:-1:2][prob > rand] = v[i+1::2][prob > rand]
        v[i+1::2][prob > rand] = v_copy
        h_copy = np.copy(h[i:-1:2][prob > rand])
        h[i:-1:2][prob > rand] = h[i+1::2][prob > rand]
        h[i+1::2][prob > rand] = h_copy
        self.ex_iter += 1
