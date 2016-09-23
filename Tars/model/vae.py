import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from progressbar import ProgressBar
from lasagne.updates import total_norm_constraint

from ..utils import (
    gauss_unitgauss_kl,
    t_repeat,
    log_mean_exp,
)
from ..distribution import UnitGaussian


class VAE(object):

    def __init__(self, q, p, n_batch, optimizer,
                 l=1, k=1, alpha=None, random=1234):
        self.q = q
        self.p = p
        self.n_batch = n_batch
        self.optimizer = optimizer
        self.l = l
        self.k = k
        self.alpha = alpha

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
        annealing_beta = T.fscalar("beta")

        mean, var = self.q.fprop(x, self.srng, deterministic=False)
        kl = gauss_unitgauss_kl(mean, var).mean()
        rep_x = [t_repeat(_x, self.l, axis=0) for _x in x]
        z = self.q.sample_given_x(rep_x, self.srng, deterministic=False)

        inverse_z = self.inverse_samples(z)
        loglike = self.p.log_likelihood_given_x(inverse_z, deterministic=False).mean()

        lowerbound = [-kl, loglike]
        loss = -(loglike-annealing_beta*kl)

        q_params = self.q.get_params()
        p_params = self.p.get_params()
        params = q_params + p_params

        grads = T.grad(loss, params)
        updates = self.optimizer(grads, params)
        self.lowerbound_train = theano.function(
            inputs=x+[annealing_beta],
            outputs=lowerbound,
            updates=updates,
            on_unused_input='ignore')

    def lowerbound_renyi(self, alpha):
        x = self.q.inputs
        rep_x = [t_repeat(_x, self.l, axis=0) for _x in x]
        q_samples = self.q.sample_given_x(
            rep_x, self.srng, deterministic=False)
        log_iw = self.log_importance_weight(q_samples, deterministic=False)
        log_iw_matrix = log_iw.reshape((x[0].shape[0], self.k))

        if alpha == 1:
            log_likelihood = T.mean(
                log_iw_matrix, axis=1)
            
        elif alpha == -np.inf:
            log_likelihood = T.max(
                log_iw_matrix, axis=1)

        else:
            log_iw_matrix = log_iw_matrix * (1 - alpha)
            log_likelihood = log_mean_exp(
                log_iw_matrix, axis=1, keepdims=True) / (1 - alpha)

        log_likelihood = T.mean(log_likelihood)

        q_params = self.q.get_params()
        p_params = self.p.get_params()
        params = q_params + p_params

        grads = T.grad(-log_likelihood, params)
        updates = self.optimizer(grads, params)
        self.lowerbound_train = theano.function(
            inputs=x, outputs=log_likelihood,
            updates=updates,
            on_unused_input='ignore')

    def train(self, train_set, annealing_beta=1):
        n_x = train_set[0].shape[0]
        nbatches = n_x // self.n_batch
        lowerbound_train = []

        for i in range(nbatches):
            start = i * self.n_batch
            end = start + self.n_batch

            batch_x = [_x[start:end] for _x in train_set]

            if self.alpha is None:
                train_L = self.lowerbound_train(*batch_x+[annealing_beta])
            else:
                train_L = self.lowerbound_train(*batch_x)
                if train_L == np.inf:
                    sys.exit()

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

        n_x = test_set[0].shape[0]
        nbatches = n_x // self.n_batch

        pbar = ProgressBar(maxval=nbatches).start()
        all_log_likelihood = []
        for i in range(nbatches):
            start = i * self.n_batch
            end = start + self.n_batch
            batch_x = [_x[start:end] for _x in test_set]
            log_likelihood = get_log_likelihood(*batch_x)
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
        kl = 0.5 * T.sum(1 + T.log(var) - mean**2 - var, axis=1)

        samples = self.q.sample_given_x(rep_x, self.srng, deterministic=True)

        inverse_samples = self.inverse_samples(samples)
        log_iw = self.p.log_likelihood_given_x(inverse_samples, deterministic=True)
        log_iw_matrix = T.reshape(log_iw, (n_x, l))
        log_marginal_estimate = kl + T.mean(log_iw_matrix, axis=1)

        return log_marginal_estimate

    def log_marginal_likelihood_iwae(self, x, k):
        n_x = x[0].shape[0]
        rep_x = [t_repeat(_x, k, axis=0) for _x in x]
        samples = self.q.sample_given_x(rep_x, self.srng, deterministic=True)

        log_iw = self.log_importance_weight(samples, deterministic=True)
        log_iw_matrix = T.reshape(log_iw, (n_x, k))
        log_marginal_estimate = log_mean_exp(
            log_iw_matrix, axis=1, keepdims=True)

        return log_marginal_estimate

    def log_importance_weight(self, samples, deterministic=False):
        """
        inputs : [[x,y,...],z1,z2,...,zn]
        outputs : log p(x,z1,z2,...,zn|y,...)/q(z1,z2,...,zn|x,y,...)
        """
        log_iw = 0

        """
        log q(z1,z2,...,zn|x,y,...)
        samples : [[x,y,...],z1,z2,...,zn]
        """
        q_log_likelihood = self.q.log_likelihood_given_x(samples, deterministic=deterministic)

        """
        log p(x|z1,z2,...,zn,y,...)
        inverse_samples : [[zn,y,...],zn-1,...,x]
        """
        inverse_samples = self.inverse_samples(samples, deterministic=deterministic)
        p_log_likelihood = self.p.log_likelihood_given_x(inverse_samples, deterministic=deterministic)

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
