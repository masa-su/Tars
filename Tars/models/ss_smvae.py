from copy import copy, deepcopy

import theano
import theano.tensor as T
import lasagne
import numpy as np
from progressbar import ProgressBar

from ..utils import tolist, log_mean_exp
from ss_hmvae import SS_HMVAE


class SS_SMVAE(SS_HMVAE):

    def __init__(self, q, p, f, s_q, prior=None,
                 n_batch=100, n_batch_u=100,
                 optimizer=lasagne.updates.adam,
                 optimizer_params={},
                 clip_grad=None, max_norm_constraint=None,
                 seed=1234):

        super(SS_SMVAE,
              self).__init__(q, p, f, s_q, prior,
                             n_batch=n_batch,
                             n_batch_u=n_batch_u,
                             optimizer=optimizer,
                             optimizer_params=optimizer_params,
                             clip_grad=clip_grad,
                             max_norm_constraint=max_norm_constraint,
                             seed=seed)

    def _discriminate(self, x, y, deterministic=False, l=1):
        # Note: we implement only in the case that the number of sampling a is 1.
        # a ~ q(a|x0,...,xn)
        a = self.q.sample_given_x(x,
                                  deterministic=deterministic)[-1:]
        # x <= [x0, x1, a]
        x = x + a
        
        # log q(y|x0,...xn, a)
        # _samples : [[x0,...,xn, a],y]
        _samples = [x] + y
        log_likelihood = self.f.log_likelihood_given_x(
            _samples, deterministic=deterministic)
        loss = -T.mean(log_likelihood)
        params = self.f.get_params() + self.q.get_params()

        return log_likelihood, loss, params

    def _vr_bound(self, x, l=1, k=1,
                  iw_alpha=0, deterministic=False, y=None):
        if y is None:
            supervised = False
        else:
            supervised = True
        rep_x = [T.extra_ops.repeat(_x, l * k, axis=0) for _x in x]

        if supervised is False:
            # a ~ q(a|x)
            rep_a = self.q.sample_given_x(rep_x,
                                          deterministic=deterministic)[-1:]
            # y ~ q(y|x,a)
            rep_y = self.f.sample_given_x(rep_x + rep_a,
                                          deterministic=deterministic)[-1:]
        else:
            rep_y = [T.extra_ops.repeat(_y, l * k,
                                        axis=0) for _y in y]

        # x,a1,..,aL ~ q(a|x)
        a = self.q.sample_given_x(rep_x, deterministic=deterministic)
        # z ~ q(z|a,y,x)
        z = self.s_q.sample_given_x(a[-1:]+rep_y+rep_x,
                                    deterministic=deterministic)[-1:]
        # q_samples = [[x0,...,xn],a1,...,aL,z]
        q_samples = a+z

        # importance weighted
        log_iw = self._log_importance_weight(q_samples, rep_y,
                                             supervised=supervised,
                                             deterministic=deterministic)

        log_iw_matrix = log_iw.reshape((x[0].shape[0] * l, k))
        log_likelihood = log_mean_exp(
            log_iw_matrix, axis=1, keepdims=True)

        log_likelihood = log_likelihood.reshape((x[0].shape[0], l))
        log_likelihood = T.mean(log_likelihood, axis=1)
        loss = -T.mean(log_likelihood)

        p_params = []
        for i, p in enumerate(self.p):
            p_params += p.get_params()
        q_params = self.q.get_params()
        q_params += self.s_q.get_params()
        params = q_params + p_params

        if supervised is False:
            params += self.f.get_params()

        if self.prior_mode == "MultiPrior":
            params += self.prior.get_params()

        params = sorted(set(params), key=params.index)
        return log_likelihood, loss, params

    def _log_importance_weight(self, samples, y, supervised=True,
                               deterministic=True):
        """
        Paramaters
        ----------
        samples : list
           [[x0,..,xn],a1,a2,...,aL,z]
        samples : y
           [y]
        Returns
        -------
        log_iw : array, shape (n_samples)
           Estimated log likelihood.
           supervised=True:
             log p(x0,...xn,a1,a2,...,aL,y,z)/q(a1,a2,...,aL,y,z|x0,...xn)
           supervised=False:
             log p(x0,...xn,a1,a2,...,aL,y,z)/q(a1,a2,...,aL,z|x0,...xn,y)
        """

        # samples : [[x0,..,xn],a1,a2,...,aL,z]
        log_iw = 0

        # log q(a1,a2,...,aL|x0,...xn)
        # _samples : [[x0,..,xn],a1,a2,...,aL]
        _samples = samples[:-1]
        q_log_likelihood = self.q.log_likelihood_given_x(
            _samples, deterministic=deterministic)

        # log q(z|aL,y, x0,...,xn)
        # _samples : [[aL,y,x0,...,xn],z]
        _samples = [tolist(samples[-2])+y+samples[0], samples[-1]]
        q_log_likelihood += self.s_q.log_likelihood_given_x(
            _samples, deterministic=deterministic)

        if supervised is False:
            # a ~ q(a|x)
            rep_a = self.q.sample_given_x(samples[0],
                                          deterministic=deterministic)[-1:]
            # log q(y|x0,...xn,a)
            # _samples : [[x0,...,xn,a],y]
            _samples = [samples[0]+rep_a]+y
            q_log_likelihood += self.f.log_likelihood_given_x(
                _samples, deterministic=deterministic)

        # log p(x0,...,xn|a1, z, y)
        # samples : [[x1,...,xn],a1...,aL,z]
        # _p_sample = [a1, z, y]

        p_log_likelihood_all = []
        _p_sample = [samples[1]]+samples[-1:]+y
        for i, p in enumerate(self.p):
            # p_samples : [[a1, z, y], x]
            p_samples = [_p_sample, samples[0][i]]
            p_log_likelihood = self.p[i].log_likelihood_given_x(
                p_samples, deterministic=deterministic)
            p_log_likelihood_all.append(p_log_likelihood)
        log_iw += sum(p_log_likelihood_all) - q_log_likelihood

        # log p(a1,..,aL|z,y)
        # prior_samples : [[z],aL,...,a1]
        _, prior_samples = self._inverse_samples(
            self._select_input(samples, [0]), return_prior=True)

        prior_samples = self._select_input(prior_samples,
                                           inputs=prior_samples[0]+y)

        if self.prior_mode == "MultiPrior":
            log_iw += self.prior.log_likelihood_given_x(prior_samples)
        else:
            log_iw += self.prior.log_likelihood(prior_samples)
        #log_iw += 1 / 2.  # Bernoulli prior distribution

        return log_iw
