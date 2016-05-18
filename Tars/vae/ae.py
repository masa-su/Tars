import numpy as np
import theano
import theano.tensor as T
from ..distribution import UnitGaussian
from progressbar import ProgressBar


class AE(object):

    def __init__(self, q, p, n_batch, optimizer, random=1234):
        self.q = q
        self.p = p
        self.n_batch = n_batch
        self.optimizer = optimizer

        self.p_sample_mean_given_x()
        self.q_sample_mean_given_x()

        self.lowerbound(random)

    def lowerbound(self, random):
        np.random.seed(random)
        self.srng = RandomStreams(seed=random)

        x = self.q.inputs
        z, q_param = self.q.mean(x, training=True)
        loglike, p_param = self.p.log_likelihood_given_x(x, z)
        error = -loglike

        params = q_param + p_param
        loss = T.mean(error)

        optimizer = self.optimizer(params=params)
        gparams = [T.grad(loss, param) for param in params]
        updates = optimizer.updates(gparams)
        self.lowerbound_train = theano.function(
            inputs=[x], outputs=loss, updates=updates, on_unused_input='ignore')

    def train(self, train_set_x):
        N = train_set_x.shape[0]
        nbatches = N // self.n_batch
        lowerbound_train = []

        for i in range(nbatches):
            start = i * self.n_batch
            end = start + self.n_batch

            x = train_set_x[start:end]
            train_L = self.lowerbound_train(x=x)
            lowerbound_train.append(np.array(train_L))
        lowerbound_train = np.mean(lowerbound_train, axis=0)

        return lowerbound_train

    def log_likelihood_test(self, test_set_x):
        x = self.q.inputs
        log_likelihood = self.log_marginal_likelihood(x)
        get_log_likelihood = theano.function(
            inputs=[x], outputs=log_likelihood, on_unused_input='ignore')

        N = test_set_x.shape[0]
        nbatches = N // self.n_batch

        pbar = ProgressBar(maxval=nbatches).start()
        all_loss = []
        for i in range(nbatches):
            start = i * self.n_batch
            end = start + self.n_batch
            x = test_set_x[start:end]
            loss = -get_log_likelihood(x=x)
            all_loss = np.r_[all_loss, loss]
            pbar.update(i)

        return all_loss

    def p_sample_mean_given_x(self):
        x = self.p.inputs
        samples = self.p.sample_mean_given_x(x, False)
        self.p_sample_mean_x = theano.function(
            inputs=[x], outputs=samples, on_unused_input='ignore')

    def q_sample_mean_given_x(self):
        x = self.q.inputs
        samples = self.q.sample_mean_given_x(x, False)
        self.q_sample_mean_x = theano.function(
            inputs=[x], outputs=samples, on_unused_input='ignore')

    def log_marginal_likelihood(self, x):
        n_x = x.shape[0]
        z, _ = self.q.mean(x, training=True)
        log_marginal_estimate, _ = self.p.log_likelihood_given_x(x, z)

        return log_marginal_estimate
