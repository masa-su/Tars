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

        np.random.seed(random)
        self.srng = RandomStreams(seed=random)

        self.p_sample_mean_given_x()
        self.q_sample_mean_given_x()
        self.prior = UnitGaussian()

        if alpha is None:
            self.lowerbound()
        else:
            self.lowerbound_renyi(alpha)

    def lowerbound(self):
        x = self.q.inputs
        mean, var = self.q.fprop(x, self.srng, deterministic=False)
        KL = 0.5 * T.mean(T.sum(1 + T.log(var) - mean**2 - var, axis=1))
        rep_x = [t_repeat(_x, self.l, axis=0) for _x in x]
        z = self.q.sample_given_x(rep_x, self.srng, deterministic=False)
        
        inverse_z = self.inverse_samples(z) 
        loglike = self.p.log_likelihood_given_x(inverse_z)
        loglike = T.mean(loglike)

        lowerbound = [KL, loglike]
        loss = -np.sum(lowerbound)

        q_params = self.q.get_params()
        p_params = self.p.get_params()
        params = q_params + p_params

        updates = self.optimizer(loss, params)
        self.lowerbound_train = theano.function(
            inputs=x, outputs=lowerbound, updates=updates, on_unused_input='ignore')

    def lowerbound_renyi(self, alpha):
        x = self.q.inputs
        rep_x = [t_repeat(_x, self.l, axis=0) for _x in x]
        q_samples = self.q.sample_given_x(
            rep_x, self.srng, deterministic=False)
        log_iw = self.log_importance_weight(q_samples)

        log_iw_matrix = log_iw.reshape((x[0].shape[0], self.k))
        log_iw_minus_max = log_iw_matrix - \
            T.max(log_iw_matrix, axis=1, keepdims=True)
        iw = T.exp(log_iw_minus_max)
        iw = iw**(1 - alpha)
        # (x[0].shape[0],k)
        iw_normalized = iw / T.sum(iw, axis=1, keepdims=True)

        lowerbound = T.mean(log_iw)

        q_params = self.q.get_params()
        p_params = self.p.get_params()
        params = q_params + p_params

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
        samples = self.p.sample_mean_given_x(x, self.srng, deterministic=True)
        self.p_sample_mean_x = theano.function(
            inputs=x, outputs=samples[-1], on_unused_input='ignore')

        samples = self.p.sample_given_x(x, self.srng, deterministic=True)
        self.p_sample_x = theano.function(
            inputs=x, outputs=samples[-1], on_unused_input='ignore')

        samples = self.p.fprop(x, self.srng, deterministic=True)
        self.p_sample_meanvar_x = theano.function(
            inputs=x, outputs=samples, on_unused_input='ignore')

    def q_sample_mean_given_x(self):
        x = self.q.inputs
        samples = self.q.sample_mean_given_x(x, self.srng, deterministic=True)
        self.q_sample_mean_x = theano.function(
            inputs=x, outputs=samples[-1], on_unused_input='ignore')

        samples = self.q.sample_given_x(x, self.srng, deterministic=True)
        self.q_sample_x = theano.function(
            inputs=x, outputs=samples[-1], on_unused_input='ignore')

        samples = self.q.fprop(x, self.srng, deterministic=True)
        self.q_sample_meanvar_x = theano.function(
            inputs=x, outputs=samples, on_unused_input='ignore')

    def log_marginal_likelihood(self, x, l):
        n_x = x[0].shape[0]
        rep_x = [t_repeat(_x, l, axis=0) for _x in x]

        mean, var = self.q.fprop(x, self.srng, deterministic=True)
        KL = 0.5 * T.sum(1 + T.log(var) - mean**2 - var, axis=1)

        samples = self.q.sample_given_x(rep_x, self.srng)
        
        inverse_samples = self.inverse_samples(samples)
        log_iw = self.p.log_likelihood_given_x(inverse_samples)
        log_iw_matrix = T.reshape(log_iw, (n_x, l))
        log_marginal_estimate = KL + T.mean(log_iw_matrix, axis=1)

        return log_marginal_estimate

    def log_marginal_likelihood_iwae(self, x, k):
        n_x = x[0].shape[0]
        rep_x = [t_repeat(_x, k, axis=0) for _x in x]
        samples = self.q.sample_given_x(rep_x, self.srng)

        log_iw = self.log_importance_weight(samples)
        log_iw_matrix = T.reshape(log_iw, (n_x, k))
        log_marginal_estimate = LogMeanExp(
            log_iw_matrix, axis=1, keepdims=True)

        return log_marginal_estimate

    def log_importance_weight(self, samples):
        """
        inputs : [[x,y,...],z1,z2,...,zn]
        outputs : log p(x,z1,z2,...,zn|y,...)/q(z1,z2,...,zn|x,y,...)
        """
        log_iw = 0

        """
        log q(z1,z2,...,zn|x,y,...)
        samples : [[x,y,...],z1,z2,...,zn]
        """
        q_log_likelihood = self.q.log_likelihood_given_x(samples)

        """
        log p(x|z1,z2,...,zn,y,...)
        inverse_samples : [[zn,y,...],zn-1,...,x]
        """
        inverse_samples = self.inverse_samples(samples)
        p_log_likelihood = self.p.log_likelihood_given_x(inverse_samples)

        log_iw += p_log_likelihood - q_log_likelihood
        log_iw += self.prior.log_likelihood(samples[-1])

        return log_iw

    def inverse_samples(self, samples):
        """
        inputs : [[x,y],z1,z2,...zn]
        outputs : [[zn,y],zn-1,...x]
        """
        inverse_samples = samples[::-1]
        inverse_samples[0] = [inverse_samples[0]] + inverse_samples[-1][1:]
        inverse_samples[-1] = inverse_samples[-1][0]
        return inverse_samples
