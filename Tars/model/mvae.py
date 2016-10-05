import numpy as np
import theano
import theano.tensor as T
from progressbar import ProgressBar

from . import MVAE_OLD
from ..utils import (
    gauss_gauss_kl,
    gauss_unitgauss_kl,
    t_repeat,
    log_mean_exp,
    tolist,
)


class MVAE(MVAE_OLD):

    def __init__(self, q, p, pq, n_batch, optimizer,
                 l=1, k=1, random=1234, gamma=1):
        self.gamma = gamma
        self.pq = pq
        super(MVAE, self).__init__(q, p, n_batch, optimizer,
                                   l, k, random)

    def lowerbound(self):
        x = self.q.inputs
        annealing_beta = T.fscalar("beta")

        mean, var = self.q.fprop(x, deterministic=False)
        kl = gauss_unitgauss_kl(mean, var).mean()
        z = self.q.sample_given_x(
            x, self.srng, repeat=self.l, deterministic=False)

        inverse_z = self.inverse_samples(self.single_input(z, 0))
        loglike0 = self.p[0].log_likelihood_given_x(
            inverse_z, deterministic=False).mean()

        inverse_z = self.inverse_samples(self.single_input(z, 1))
        loglike1 = self.p[1].log_likelihood_given_x(
            inverse_z, deterministic=False).mean()

        # ---penalty
        mean, var = self.q.fprop(x, deterministic=False)
        # z ~ q(x0)
        mean0, var0 = self.pq[0].fprop([x[0]], self.srng, deterministic=False)
        mean1, var1 = self.pq[1].fprop([x[1]], self.srng, deterministic=False)

        # kl[q(x0,0)||q(x0,x1)]
        kl_0 = gauss_gauss_kl(mean, var, mean0, var0).mean()
        kl_1 = gauss_gauss_kl(mean, var, mean1, var1).mean()

        # ---
        q_params = self.q.get_params()
        p0_params = self.p[0].get_params()
        p1_params = self.p[1].get_params()
        pq0_params = self.pq[0].get_params()
        pq1_params = self.pq[1].get_params()

        params = q_params + p0_params + p1_params + pq0_params + pq1_params
        lowerbound = [-kl, loglike0, loglike1, kl_0, kl_1]
        loss = annealing_beta * kl - np.sum(
            lowerbound[1:3]) + self.gamma * np.sum(lowerbound[3:])

        updates = self.optimizer(loss, params)
        self.lowerbound_train = theano.function(
            inputs=x + [annealing_beta],
            outputs=lowerbound,
            updates=updates,
            on_unused_input='ignore')

    def pq_sample_mean_given_x(self):
        x = self.pq[0].inputs
        samples = self.pq[0].sample_mean_given_x(x, deterministic=True)
        self.pq0_sample_mean_x = theano.function(
            inputs=x, outputs=samples[-1], on_unused_input='ignore')

        samples = self.pq[0].sample_given_x(x, self.srng, deterministic=True)
        self.pq0_sample_x = theano.function(
            inputs=x, outputs=samples[-1], on_unused_input='ignore')

        samples = self.pq[0].fprop(x, self.srng, deterministic=True)
        self.pq0_sample_meanvar_x = theano.function(
            inputs=x, outputs=samples, on_unused_input='ignore')

        x = self.pq[1].inputs
        samples = self.pq[1].sample_mean_given_x(x, deterministic=True)
        self.pq1_sample_mean_x = theano.function(
            inputs=x, outputs=samples[-1], on_unused_input='ignore')

        samples = self.pq[1].sample_given_x(x, self.srng, deterministic=True)
        self.pq1_sample_x = theano.function(
            inputs=x, outputs=samples[-1], on_unused_input='ignore')

        samples = self.pq[1].fprop(x, self.srng, deterministic=True)
        self.pq1_sample_meanvar_x = theano.function(
            inputs=x, outputs=samples, on_unused_input='ignore')

    def log_pseudo_mg_importance_weight(self, samples, deterministic=True):
        """
        Paramaters
        ----------
        samples : list
           [[x0],z1,z2,...,zn]

        Returns
        -------
        log_iw : array, shape (n_samples*k)
           Estimated log likelihood.
           log p(x0,z1,z2,...,zn)/q(z1,z2,...,zn|x0)
        """

        log_iw = 0

        # log q(z1,z2,...,zn|x0)
        # samples : [[x0],z1,z2,...,zn]
        q0_log_likelihood = self.pq[0].log_likelihood_given_x(
            samples, deterministic=deterministic)

        # log p(x0|z1,z2,...,zn)
        # inverse_samples0 : [zn,zn-1,...,x0]
        inverse_samples0 = self.inverse_samples(self.single_input(samples, 0))
        p0_log_likelihood = self.p[0].log_likelihood_given_x(
            inverse_samples0, deterministic=deterministic)

        log_iw += p0_log_likelihood - q0_log_likelihood
        log_iw += self.prior.log_likelihood(samples[-1])

        return log_iw

    def log_pseudo_conditional_importance_weight(self, samples,
                                                 deterministic=True):
        """
        Paramaters
        ----------
        samples : list
           [[x0,x1],z1,z2,...,zn]

        Returns
        -------
        log_iw : array, shape (n_samples*k)
           Estimated log likelihood.
           log p(x0,x1,z1,z2,...,zn)
               /q(z1,z2,...,zn|x1)p(x1)
        """

        # log q(z1,z2,...,zn|x0,x1)
        # samples_x1 : [[x1],z1,z2,...,zn]
        samples_x1 = self.single_input(samples, 1)
        q_log_likelihood = self.pq[1].log_likelihood_given_x(
            samples_x1, deterministic=deterministic)

        # log p(x0|z1,z2,...,zn)
        # inverse_samples0 : [zn,zn-1,...,x0]
        inverse_samples0 = self.inverse_samples(self.single_input(samples, 0))
        p0_log_likelihood = self.p[0].log_likelihood_given_x(
            inverse_samples0, deterministic=deterministic)

        # log p(x1|z1,z2,...,zn)
        # inverse_samples1 : [zn,zn-1,...,x1]
        inverse_samples1 = self.inverse_samples(self.single_input(samples, 1))
        p1_log_likelihood = self.p[1].log_likelihood_given_x(
            inverse_samples1, deterministic=deterministic)

        # log p(x1) = logmeanexp(log p(w|z)), where z~N(0,1)
        # samples : [std_z,x1] TODO: multiple latent variable
        z = inverse_samples1[0][0]
        x1 = inverse_samples1[-1]
        # single sampling
        std_z = self.prior.sample(z.shape, self.srng)
        p1_mg_log_likelihood = self.p[1].log_likelihood_given_x(
            [[std_z], x1], deterministic=deterministic)

        log_iw = p0_log_likelihood + p1_log_likelihood - \
            q_log_likelihood - p1_mg_log_likelihood

        log_iw += self.prior.log_likelihood(samples[-1])

        return log_iw

    def log_likelihood_iwae(self, x, k, type_p="joint"):
        """
        Paramaters
        ----------
        x : TODO

        k : TODO

        type_p : {'joint', 'conditional', 'marginal' 'pseudo_marginal'}
           Specifies the type of the log likelihood.

        Returns
        --------
        log_marginal_estimate : array, shape (n_samples)
           Estimated log likelihood.

        """

        n_x = x[0].shape[0]
        rep_x = [t_repeat(_x, k, axis=0) for _x in x]

        if type_p == "pseudo_marginal":
            samples = self.pq[0].sample_given_x(
                rep_x, self.srng, deterministic=True)
            log_iw = self.log_pseudo_mg_importance_weight(
                samples)
        elif type_p == "pseudo_conditional":
            samples = self.pq[1].sample_given_x(
                tolist(rep_x[1]), self.srng, deterministic=True)
            samples[0] = rep_x
            log_iw = self.log_pseudo_conditional_importance_weight(
                samples)
        else:
            samples = self.q.sample_given_x(
                rep_x, self.srng, deterministic=True)
            if type_p == "joint":
                log_iw = self.log_importance_weight(
                    samples)
            elif type_p == "marginal":
                log_iw = self.log_mg_importance_weight(
                    samples)
            elif type_p == "conditional":
                log_iw = self.log_conditional_importance_weight(
                    samples)

        log_iw_matrix = T.reshape(log_iw, (n_x, k))
        log_marginal_estimate = log_mean_exp(
            log_iw_matrix, axis=1, keepdims=True)

        return log_marginal_estimate

    def log_likelihood_test(self, test_set, l=1, k=1,
                            mode='iw', type_p="joint", n_batch=None):
        """
        Paramaters
        ----------
        test_set : TODO

        l : TODO

        k : TODO

        mode : {'iw', 'lower_bound'}
           Specifies the way of sampling to estimate the log likelihood,
           whether an importance weighted lower bound or a original
           lower bound.

        type_p : {'joint', 'conditional', 'marginal' 'pseudo_marginal'}
           Specifies the type of the log likelihood.

        Returns
        --------
        all_log_likelihood : array, shape (n_samples)
           Estimated log likelihood.

        """

        if n_batch is None:
            n_batch = self.n_batch

        if mode not in ['iw', 'lower_bound']:
            raise ValueError("mode must be whether 'iw' or 'lower_bound',"
                             "got %s." % mode)

        if type_p not in ['joint', 'conditional', 'marginal',
                          'pseudo_marginal', 'pseudo_conditional']:
            raise ValueError("type_p must be one of {'joint', 'conditional', "
                             "'marginal', 'pseudo_marginal', "
                             "'pseudo_conditional'}, got %s." % type_p)

        x = self.q.inputs
        if type_p == "pseudo_marginal":
            x = tolist(x[0])

        log_likelihood = self.log_likelihood_iwae(x, k, type_p=type_p)
        get_log_likelihood = theano.function(
            inputs=x, outputs=log_likelihood, on_unused_input='ignore')

        print "start sampling"

        n_x = test_set[0].shape[0]
        nbatches = n_x // n_batch

        pbar = ProgressBar(maxval=nbatches).start()
        all_log_likelihood = []
        for i in range(nbatches):
            start = i * n_batch
            end = start + n_batch
            batch_x = [_x[start:end] for _x in test_set]
            log_likelihood = get_log_likelihood(*batch_x)
            all_log_likelihood = np.r_[all_log_likelihood, log_likelihood]
            pbar.update(i)

        return all_log_likelihood
