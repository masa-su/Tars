import numpy as np
import theano
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from progressbar import ProgressBar
import six


class AE(object):

    def __init__(self, q, p, n_batch, optimizer, random=1234):
        self.q = q
        self.p = p
        self.n_batch = n_batch
        self.optimizer = optimizer

        np.random.seed(random)
        self.srng = RandomStreams(seed=random)

        self.p_sample_mean_x = self._init_p_sample_mean_given_x()
        self.q_sample_mean_x = self._init_q_sample_mean_given_x()

        self.lowerbound_train = self._init_lowerbound()

    def _init_lowerbound(self):
        x = self.q.inputs
        z = self.q.fprop(x, deterministic=False)
        inverse_z = self.inverse_samples([x, z])
        loglike = self.p.log_likelihood_given_x(inverse_z).mean()

        q_params = self.q.get_params()
        p_params = self.p.get_params()
        params = q_params + p_params

        updates = self.optimizer(-loglike, params)
        return theano.function(
            inputs=x,
            outputs=loglike,
            updates=updates,
            on_unused_input='ignore')

    def train(self, train_set):
        n_x = train_set[0].shape[0]
        nbatches = n_x // self.n_batch
        lowerbound_train = []

        for i in six.moves.range(nbatches):
            start = i * self.n_batch
            end = start + self.n_batch

            batch_x = [_x[start:end] for _x in train_set]
            train_L = self.lowerbound_train(*batch_x)
            lowerbound_train.append(np.array(train_L))
        lowerbound_train = np.mean(lowerbound_train, axis=0)

        return lowerbound_train

    def log_likelihood_test(self, test_set):
        x = self.q.inputs
        log_likelihood = self.log_marginal_likelihood(x)
        get_log_likelihood = theano.function(
            inputs=x, outputs=log_likelihood, on_unused_input='ignore')

        n_x = test_set[0].shape[0]
        nbatches = n_x // self.n_batch

        pbar = ProgressBar(maxval=nbatches).start()
        all_log_likelihood = []
        for i in six.moves.range(nbatches):
            start = i * self.n_batch
            end = start + self.n_batch
            batch_x = [_x[start:end] for _x in test_set]
            log_likelihood = get_log_likelihood(*batch_x)
            all_log_likelihood = np.r_[all_log_likelihood, log_likelihood]
            pbar.update(i)

        return all_log_likelihood

    def _init_p_sample_mean_given_x(self):
        x = self.p.inputs
        samples = self.p.sample_mean_given_x(x, False)
        return theano.function(
            inputs=x, outputs=samples[-1], on_unused_input='ignore')

    def _init_q_sample_mean_given_x(self):
        x = self.q.inputs
        samples = self.q.sample_mean_given_x(x, False)
        return theano.function(
            inputs=x, outputs=samples[-1], on_unused_input='ignore')

    def log_marginal_likelihood(self, x):
        z = self.q.fprop(x, deterministic=True)
        inverse_z = self.inverse_samples([x, z])
        log_marginal_estimate = self.p.log_likelihood_given_x(inverse_z)

        return log_marginal_estimate

    def inverse_samples(self, samples):
        """
        inputs : [[x,y],z1,z2,...zn]
        outputs : [[zn,y],zn-1,...x]
        """
        inverse_samples = samples[::-1]
        inverse_samples[0] = [inverse_samples[0]] + inverse_samples[-1][1:]
        inverse_samples[-1] = inverse_samples[-1][0]
        return inverse_samples
