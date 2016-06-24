from Tars.model import VAE
import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from progressbar import ProgressBar
from ..util import KL_gauss_gauss, KL_gauss_unitgauss, t_repeat, LogMeanExp
from ..distribution import UnitGaussian
from copy import copy

class MVAE(VAE):

    def __init__(self, q, p, n_batch, optimizer, l=1, k=1, alpha=None, random=1234, gamma=0.1):
        self.gamma = gamma
        super(MVAE, self).__init__(q, p, n_batch, optimizer, l, k, None, random)
        self.pq_sample_mean_given_x()

    def lowerbound(self):
        x = self.q.inputs
        mean, var = self.q.fprop(x, deterministic=False)
        KL = KL_gauss_unitgauss(mean, var).mean()
        rep_x = [t_repeat(_x, self.l, axis=0) for _x in x]
        z = self.q.sample_given_x(rep_x, self.srng, deterministic=False)

        inverse_z = self.inverse_samples(self.single_input(z,0))
        loglike0 = self.p[0].log_likelihood_given_x(inverse_z).mean()

        inverse_z = self.inverse_samples(self.single_input(z,1))
        loglike1 = self.p[1].log_likelihood_given_x(inverse_z).mean()
        
        # ---penalty
        mean, var = self.q.fprop(x, deterministic=False)
        # z ~ q(x0)
        mean0, var0 = self.q.fprop([x[0],T.zeros_like(x[1])], self.srng, deterministic=False)
        mean1, var1 = self.q.fprop([T.zeros_like(x[0]),x[1]], self.srng, deterministic=False)

        # KL[q(x0,0)||q(x0,x1)]
        KL_0 = KL_gauss_gauss(mean, var, mean0, var0).mean()
        KL_1 = KL_gauss_gauss(mean, var, mean1, var1).mean()

        # ---

        q_params = self.q.get_params()
        p0_params = self.p[0].get_params()
        p1_params = self.p[1].get_params()
        params = q_params + p0_params + p1_params
        lowerbound = [-KL, loglike0, loglike1, KL_0, KL_1]
        loss = -np.sum(lowerbound[:3])+self.gamma*np.sum(lowerbound[3:])

        updates = self.optimizer(loss, params)
        self.lowerbound_train = theano.function(
            inputs=x, outputs=lowerbound, updates=updates, on_unused_input='ignore')

    def p_sample_mean_given_x(self):
        x = self.p[0].inputs
        samples = self.p[0].sample_mean_given_x(x, deterministic=True)
        self.p0_sample_mean_x = theano.function(
            inputs=x, outputs=samples[-1], on_unused_input='ignore')

        samples = self.p[0].sample_given_x(x, self.srng, deterministic=True)
        self.p0_sample_x = theano.function(
            inputs=x, outputs=samples[-1], on_unused_input='ignore')

        x = self.p[1].inputs
        samples = self.p[1].sample_mean_given_x(x, deterministic=True)
        self.p1_sample_mean_x = theano.function(
            inputs=x, outputs=samples[-1], on_unused_input='ignore')

        samples = self.p[1].sample_given_x(x, self.srng, deterministic=True)
        self.p1_sample_x = theano.function(
            inputs=x, outputs=samples[-1], on_unused_input='ignore')

    def pq_sample_mean_given_x(self):
        x = self.q.inputs
        samples = self.q.sample_mean_given_x([x[0],T.zeros_like(x[1])], deterministic=True)
        self.pq0_sample_mean_x = theano.function(
            inputs=x, outputs=samples[-1], on_unused_input='ignore')

        samples = self.q.sample_given_x([x[0],T.zeros_like(x[1])], self.srng, deterministic=True)
        self.pq0_sample_x = theano.function(
            inputs=x, outputs=samples[-1], on_unused_input='ignore')

        samples = self.q.sample_mean_given_x([T.zeros_like(x[0]),x[1]], deterministic=True)
        self.pq1_sample_mean_x = theano.function(
            inputs=x, outputs=samples[-1], on_unused_input='ignore')

        samples = self.q.sample_given_x([T.zeros_like(x[0]),x[1]], self.srng, deterministic=True)
        self.pq1_sample_x = theano.function(
            inputs=x, outputs=samples[-1], on_unused_input='ignore')

    def log_importance_weight(self, samples):
        """
        inputs : [[x0,x1],z1,z2,...,zn]
        outputs : log p(x0,x1,z1,z2,...,zn)/q(z1,z2,...,zn|x0,x1)
        """
        log_iw = 0

        """
        log q(z1,z2,...,zn|x0,x1)
        samples : [[x0,x1],z1,z2,...,zn]
        """
        q_log_likelihood = self.q.log_likelihood_given_x(samples)

        """
        log p(x0|z1,z2,...,zn,y,...)
        inverse_samples0 : [zn,zn-1,...,x0]
        """
        inverse_samples0 = self.inverse_samples(self.single_input(samples,0))
        p0_log_likelihood = self.p[0].log_likelihood_given_x(inverse_samples0)

        """
        log p(x1|z1,z2,...,zn,y,...)
        inverse_samples1 : [zn,zn-1,...,x1]
        """
        inverse_samples1 = self.inverse_samples(self.single_input(samples,1))
        p1_log_likelihood = self.p[1].log_likelihood_given_x(inverse_samples1)

        log_iw += p0_log_likelihood + p1_log_likelihood - q_log_likelihood
        log_iw += self.prior.log_likelihood(samples[-1])

        return log_iw

    def penalty_test(self, test_set, l=1):
        x = self.q.inputs
        rep_x = [t_repeat(_x, l, axis=0) for _x in x]

        # z ~ q(x0,random_x)
        z0 = self.q.sample_given_x([rep_x[0],T.zeros_like(rep_x[1])], self.srng, deterministic=True)
        z1 = self.q.sample_given_x([T.zeros_like(rep_x[0]),rep_x[1]], self.srng, deterministic=True)

        inverse_z0 = self.inverse_samples(self.single_input(z1,input=rep_x[0]))
        loglike0_given0 = self.p[0].log_likelihood_given_x(inverse_z0)# p(x0|z0)
        loglike0_given0 = T.mean(loglike0_given0)

        inverse_z1 = self.inverse_samples(self.single_input(z0,input=rep_x[1]))
        loglike1_given1 = self.p[1].log_likelihood_given_x(inverse_z1)# p(x1|z1)
        loglike1_given1 = T.mean(loglike1_given1)
        
        loss = [-loglike0_given0,-loglike1_given1]
        self.loss_test = theano.function(
            inputs=x, outputs=loss, on_unused_input='ignore')

        N = test_set[0].shape[0]
        nbatches = N // self.n_batch
        pbar = ProgressBar(maxval=nbatches).start()
        loss = []

        for i in range(nbatches):
            start = i * self.n_batch
            end = start + self.n_batch

            x = [_x[start:end] for _x in test_set]
            test_L = self.loss_test(*x)
            loss.append(np.array(test_L))
            pbar.update(i)
        loss = np.mean(loss, axis=0)
        return loss

    def single_input(self, samples, i=0, input=None):
        """
        inputs : [[x,y,...],z1,z2,....]
        outputs : 
           i=0 : [[x],z1,z2,....]
           i=1 : [[y],z1,z2,....]
        """
        _samples = copy(samples)
        if input:
            _samples[0] = [input]
        else:
            _samples[0] = [_samples[0][i]]
        return _samples
