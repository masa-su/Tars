import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from progressbar import ProgressBar
from ..util import t_repeat, LogMeanExp
from ..distribution import UnitGaussian


class VAE(object):

    def __init__(self, q, p, n_batch, optimizer, l=1, k=1, alpha=None, random=1234):
        self.q = q
        self.p = p
        self.n_batch = n_batch
        self.optimizer = optimizer
        self.l = l
        self.k = k

        self.p_sample_mean_given_x()
        self.q_sample_mean_given_x()
        self.prior = UnitGaussian()

        if alpha is None:
            self.lowerbound(random)
        else:
            self.lowerbound_renyi(alpha, random)

    def lowerbound(self, random):
        np.random.seed(random)
        self.srng = RandomStreams(seed=random)

        x = self.q.inputs
        mean, var, q_param = self.q.mean(x, deterministic=False)
        KL = 0.5 * T.mean(T.sum(1 + T.log(var) - mean**2 - var, axis=1))
        rep_x = [t_repeat(_x, self.l, axis=0) for _x in x]
        z, _ = self.q.sample_given_x(rep_x, self.srng, deterministic=False)

        loglike, p_param = self.p.log_likelihood_given_x(
            rep_x[0], [z] + rep_x[1:])  # p(x|z,y,...)
        loglike = T.mean(loglike)

        params = q_param + p_param
        lowerbound = [KL, loglike]
        loss = -np.sum(lowerbound)

        updates = self.optimizer(loss, params)
        self.lowerbound_train = theano.function(
            inputs=x, outputs=lowerbound, updates=updates, on_unused_input='ignore')

    def lowerbound_renyi(self, alpha, random):
        np.random.seed(random)
        self.srng = RandomStreams(seed=random)

        x = self.q.inputs
        rep_x = [t_repeat(_x, self.l, axis=0) for _x in x]
        q_samples, _ = self.q.sample_given_x(
            rep_x, self.srng, deterministic=False)
        log_iw, params = self.log_importance_weight(rep_x, q_samples)

        log_iw_matrix = log_iw.reshape((x[0].shape[0], self.k))
        log_iw_minus_max = log_iw_matrix - \
            T.max(log_iw_matrix, axis=1, keepdims=True)
        iw = T.exp(log_iw_minus_max)
        iw = iw**(1 - alpha)
        # (x[0].shape[0],k)
        iw_normalized = iw / T.sum(iw, axis=1, keepdims=True)

        lowerbound = T.mean(log_iw)

        if alpha == -np.inf:
            gparams = [T.grad(-T.sum(T.max(log_iw_matrix, axis=1)), param)
                       for param in params]

        else:
            iw_normalized_vector = T.reshape(
                iw_normalized, log_iw.shape)  # (x[0].shape[0]*num_samples)
            dummy_vec = T.vector(dtype=theano.config.floatX)
            gparams = [
                theano.clone(
                    T.grad(-T.dot(log_iw, dummy_vec), param),
                    replace={dummy_vec: iw_normalized_vector})
                for param in params]

        updates = self.optimizer(gparams, params)
        self.lowerbound_train = theano.function(
            inputs=x, outputs=lowerbound, updates=updates, on_unused_input='ignore')

    def train(self, train_set):
        N = train_set[0].shape[0]
        nbatches = N // self.n_batch
        lowerbound_train = []

        for i in range(nbatches):
            start = i * self.n_batch
            end = start + self.n_batch

            x = [_x[start:end] for _x in train_set]
            train_L = self.lowerbound_train(*x)
            lowerbound_train.append(np.array(train_L))
        lowerbound_train = np.mean(lowerbound_train, axis=0)

        return lowerbound_train

    def log_likelihood_test(self, test_set, l=1, k=1, mode='iw'):
        x = self.q.inputs
        if mode == 'iw':
            log_likelihood = self.log_marginal_likelihood_iwae(x, k)
        else:
            log_likelihood = self.log_marginal_likelihood(x, l)
        get_log_likelihood = theano.function(
            inputs=x, outputs=log_likelihood, on_unused_input='ignore')

        print "start sampling"

        N = test_set[0].shape[0]
        nbatches = N // self.n_batch

        pbar = ProgressBar(maxval=nbatches).start()
        all_log_likelihood = []
        for i in range(nbatches):
            start = i * self.n_batch
            end = start + self.n_batch
            x = [_x[start:end] for _x in test_set]
            log_likelihood = get_log_likelihood(*x)
            all_log_likelihood = np.r_[all_log_likelihood, log_likelihood]
            pbar.update(i)

        return all_log_likelihood

    def p_sample_mean_given_x(self):
        x = self.p.inputs
        samples, _ = self.p.sample_mean_given_x(x, deterministic=True)
        self.p_sample_mean_x = theano.function(
            inputs=x, outputs=samples, on_unused_input='ignore')

    def q_sample_mean_given_x(self):
        x = self.q.inputs
        samples, _ = self.q.sample_mean_given_x(x, deterministic=True)
        self.q_sample_mean_x = theano.function(
            inputs=x, outputs=samples, on_unused_input='ignore')

    def log_marginal_likelihood(self, x, l):
        n_x = x[0].shape[0]
        rep_x = [t_repeat(_x, l, axis=0) for _x in x]

        mean, var, KL_param = self.q.mean(x, deterministic=True)
        KL = 0.5 * T.sum(1 + T.log(var) - mean**2 - var, axis=1)

        samples, _ = self.q.sample_given_x(rep_x, self.srng)

        log_iw, _ = self.p.log_likelihood_given_x(rep_x, samples)
        log_iw_matrix = T.reshape(log_iw, (n_x, l))
        log_marginal_estimate = KL + T.mean(log_iw_matrix, axis=1)

        return log_marginal_estimate

    def log_marginal_likelihood_iwae(self, x, k):
        n_x = x[0].shape[0]
        rep_x = [t_repeat(_x, k, axis=0) for _x in x]
        samples, _ = self.q.sample_given_x(rep_x, self.srng)

        log_iw, _ = self.log_importance_weight(rep_x, samples)
        log_iw_matrix = T.reshape(log_iw, (n_x, k))
        log_marginal_estimate = LogMeanExp(
            log_iw_matrix, axis=1, keepdims=True)

        return log_marginal_estimate

    def log_importance_weight(self, rep_x, samples):
        log_iw = 0
        p_log_likelihood, p_param = self.p.log_likelihood_given_x(
            rep_x[0], [samples] + rep_x[1:])  # p(x|z,y,...)
        q_log_likelihood, q_param = self.q.log_likelihood_given_x(
            samples, rep_x)  # p(z|x,y,...)

        # log p(x|z) - log q(z|x)
        log_iw += p_log_likelihood - q_log_likelihood

        # log p(z)
        log_iw += self.prior.log_likelihood(samples)

        # log p(x,z)/q(z|x)
        return log_iw, q_param + p_param
