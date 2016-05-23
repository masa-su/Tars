from Tars.vae.vae import VAE
import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from progressbar import ProgressBar
from ..util import t_repeat, LogMeanExp
from ..distribution import UnitGaussian


class MVAE(VAE):

    def __init__(self, q, p, n_batch, optimizer, l=1, k=1, alpha=None, random=1234, gamma=0.1):
        self.gamma = gamma
        super(MVAE, self).__init__(q, p, n_batch, optimizer, l, k, alpha, random)

    def lowerbound(self, random):
        np.random.seed(random)
        self.srng = RandomStreams(seed=random)

        x = self.q.inputs
        mean, var, q_param = self.q.mean(x, deterministic=False)
        KL = 0.5 * T.mean(T.sum(1 + T.log(var) - mean**2 - var, axis=1))
        rep_x = [t_repeat(_x, self.l, axis=0) for _x in x]
        z, _ = self.q.sample_given_x(rep_x, self.srng, deterministic=False)

        loglike0, p0_param = self.p[0].log_likelihood_given_x(
            rep_x[0], [z])  # p(x0|z)
        loglike0 = T.mean(loglike0)

        loglike1, p1_param = self.p[1].log_likelihood_given_x(
            rep_x[1], [z])  # p(x1|z)
        loglike1 = T.mean(loglike1)

        # ---penalty TODO: evaluation
        z0, _ = self.q.sample_given_x([rep_x[0],T.zeros_like(rep_x[1])], self.srng, deterministic=False)
        z1, _ = self.q.sample_given_x([T.zeros_like(rep_x[0]),rep_x[1]], self.srng, deterministic=False)

        loglike0_given0, _ = self.p[0].log_likelihood_given_x(
            rep_x[0], [z0])  # p(x0|z0)
        loglike0_given0 = T.mean(loglike0_given0)

        loglike1_given1, _ = self.p[1].log_likelihood_given_x(
            rep_x[1], [z1])  # p(x1|z1)
        loglike1_given1 = T.mean(loglike1_given1)
        # ---

        params = q_param + p0_param + p1_param
        lowerbound = [KL, loglike0, loglike1, loglike0_given0, loglike1_given1]
        loss = -np.sum(lowerbound[:3])-self.gamma*np.sum(lowerbound[3:])

        updates = self.optimizer(loss, params)
        self.lowerbound_train = theano.function(
            inputs=x, outputs=lowerbound, updates=updates, on_unused_input='ignore')

    def p_sample_mean_given_x(self):
        x = self.p[0].inputs
        samples, _ = self.p[0].sample_mean_given_x(x, deterministic=True)
        self.p0_sample_mean_x = theano.function(
            inputs=x, outputs=samples, on_unused_input='ignore')

        x = self.p[1].inputs
        samples, _ = self.p[1].sample_mean_given_x(x, deterministic=True)
        self.p1_sample_mean_x = theano.function(
            inputs=x, outputs=samples, on_unused_input='ignore')

    def log_importance_weight(self, rep_x, samples):
        log_iw = 0
        p0_log_likelihood, p0_param = self.p[0].log_likelihood_given_x(
            rep_x[0], [samples])  # p(x0|z)
        p1_log_likelihood, p1_param = self.p[1].log_likelihood_given_x(
            rep_x[1], [samples])  # p(x1|z)
        q_log_likelihood, q_param = self.q.log_likelihood_given_x(
            samples, rep_x)  # q(z|x0,x1)

        # log p(x0,x1|z) - log q(z|x0,x1)
        log_iw += p0_log_likelihood + p1_log_likelihood - q_log_likelihood

        # log p(z)
        log_iw += self.prior.log_likelihood(samples)

        # log p(x0,x1,z)/q(z|x0,x1)
        return log_iw, q_param + p0_param + p1_param
