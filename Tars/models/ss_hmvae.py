from copy import copy, deepcopy

import theano
import theano.tensor as T
import lasagne
import numpy as np
from progressbar import ProgressBar

from ..utils import tolist, log_mean_exp
from ss_mvae import SS_MVAE


class SS_HMVAE(SS_MVAE):

    def __init__(self, q, p, f, s_q, prior=None,
                 n_batch=100, n_batch_u=100,
                 optimizer=lasagne.updates.adam,
                 optimizer_params={},
                 clip_grad=None, max_norm_constraint=None,
                 seed=1234):
        self.s_q = s_q

        super(SS_HMVAE,
              self).__init__(q, p, f, prior,
                             n_batch=n_batch, n_batch_u=n_batch_u,
                             optimizer=optimizer,
                             optimizer_params=optimizer_params,
                             clip_grad=clip_grad,
                             max_norm_constraint=max_norm_constraint,
                             seed=seed)

    def _set_train(self, l, k, *args):
        # inputs
        x_u = self.q.inputs
        x_l = deepcopy(self.q.inputs)
        y = T.fmatrix("y")

        # training
        rate = T.fscalar("rate")
        inputs = x_u + x_l + [y, l, k, rate]
        lower_bound_u, loss_u, params = self._vr_bound(x_u, l, k, 0, False)
        lower_bound_l, loss_l, _ = self._vr_bound(x_l, l, k, 0, False,
                                                  tolist(y))
        lower_bound_y, loss_y, _ = self._discriminate(x_l, tolist(y), False)

        lower_bound = [T.mean(lower_bound_u), T.mean(lower_bound_l),
                       T.mean(lower_bound_y)]

        loss = loss_u + loss_l + rate * loss_y
        updates = self._get_updates(loss, params, self.optimizer,
                                    self.optimizer_params, self.clip_grad,
                                    self.max_norm_constraint)

        self.lower_bound_train = theano.function(inputs=inputs,
                                                 outputs=lower_bound,
                                                 updates=updates,
                                                 on_unused_input='ignore')

        # training (classification)
        inputs = x_l + [y]
        lower_bound_y, loss, params = self._discriminate(x_l, tolist(y), False)
        updates = self._get_updates(loss, params, self.optimizer,
                                    self.optimizer_params, self.clip_grad,
                                    self.max_norm_constraint)

        self.classifier_train = theano.function(inputs=inputs,
                                                outputs=T.mean(lower_bound_y),
                                                updates=updates,
                                                on_unused_input='ignore')

    def _set_test(self, l, k):
        # inputs
        x_u = self.q.inputs
        x_l = deepcopy(self.q.inputs)
        y = T.fmatrix("y")

        # test
        inputs = x_u + x_l + [y, l, k]
        lower_bound_u, loss_u, _ = self._vr_bound(x_u, l, k, 0, True)
        lower_bound_l, loss_l, _ = self._vr_bound(x_l, l, k, 0, True,
                                                  tolist(y))

        lower_bound = [T.mean(lower_bound_u), T.mean(lower_bound_l)]

        self.lower_bound_test = theano.function(inputs=inputs,
                                                outputs=lower_bound,
                                                on_unused_input='ignore')

        # test (classification)
        inputs = x_l + [y]
        lower_bound_y, _, _ = self._discriminate(x_l, tolist(y), True)
        self.classifier_test = theano.function(inputs=inputs,
                                               outputs=T.mean(lower_bound_y),
                                               on_unused_input='ignore')

    def _discriminate(self, x, y, deterministic=False):
        # a ~ q(a|x0,...xn)
        a = self.q.sample_given_x(x, deterministic=deterministic)[-1:]

        # log q(y|a)
        # _samples : [[a], y]
        _samples = [a] + y
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

        # ???
        _ = self.q.sample_given_x(rep_x, deterministic=deterministic)

        # [[x],a] ~ q(a|x)
        a = self.q.sample_given_x(rep_x, deterministic=deterministic)

        if supervised is False:
            # y ~ q(y|a)
            rep_y = self.f.sample_given_x(a[-1:],
                                          deterministic=deterministic)[-1:]
        else:
            rep_y = [T.extra_ops.repeat(_y, l * k,
                                        axis=0) for _y in y]

        # z ~ q(z|a,y)
        z = self.s_q.sample_given_x(a[-1:] + rep_y,
                                    deterministic=deterministic)[-1:]
        # q_samples = [[x],a,z]
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

        # samples : [[x0,..,xn],a,z]
        log_iw = 0

        # log q(a|x0,...xn)
        # _samples : [[x0,..,xn],a]
        _samples = samples[:-1]
        q_log_likelihood = self.q.log_likelihood_given_x(
            _samples, deterministic=deterministic)

        # log q(z|a,y)
        # _samples : [[a,y],z]
        _samples = [tolist(samples[-2]) + y, samples[-1]]
        q_log_likelihood += self.s_q.log_likelihood_given_x(
            _samples, deterministic=deterministic)

        if supervised is False:
            # log q(y|a)
            # _samples : [[a],y]
            _samples = [tolist(samples[-2])] + y
            q_log_likelihood += self.f.log_likelihood_given_x(
                _samples, deterministic=deterministic)

        # log p(x0,...,xn|a,y)
        # p_samples : [[a,y],x0]
        _p_samples = tolist(samples[-2]) + y
        p_log_likelihood_all = []
        for i, p in enumerate(self.p):
            p_samples = [_p_samples, samples[0][i]]
            p_log_likelihood = self.p[i].log_likelihood_given_x(
                p_samples, deterministic=deterministic)
            p_log_likelihood_all.append(p_log_likelihood)
        log_iw += sum(p_log_likelihood_all) - q_log_likelihood

        # log p(a|z,y), log p(z)
        # prior_samples : [[z,y],a]
        prior_samples = [samples[-1:] + y, samples[-2]]

        if self.prior_mode == "MultiPrior":
            log_iw += self.prior.log_likelihood_given_x(prior_samples)
        else:
            log_iw += self.prior.log_likelihood(prior_samples)
        #log_iw += 1 / 2.  # Bernoulli prior distribution

        return log_iw
