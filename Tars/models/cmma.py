from copy import copy

import numpy as np
import theano
import theano.tensor as T
import lasagne
from progressbar import ProgressBar

from ..utils import log_mean_exp
from ..distributions.estimate_kl import analytical_kl, get_prior
from . import VAE


class CMMA(VAE):

    def __init__(self, q, p,
                 n_batch=100, optimizer=lasagne.updates.adam,
                 optimizer_params={},
                 clip_grad=None, max_norm_constraint=None,
                 train_iw=False, test_iw=True, iw_alpha=0,
                 seed=1234):
        super(CMMA,
              self).__init__(q, p, prior=None,
                             n_batch=n_batch,
                             optimizer=optimizer,
                             optimizer_params=optimizer_params,
                             clip_grad=clip_grad,
                             max_norm_constraint=max_norm_constraint,
                             train_iw=train_iw, test_iw=test_iw,
                             iw_alpha=0, seed=seed)

    def _set_test(self, type_p="normal", missing=False):
        # set inputs
        x = self.q.inputs
        l = T.iscalar("l")
        k = T.iscalar("k")

        if type_p == "normal":
            if self.test_iw:
                inputs = x + [l, k]
                lower_bound, _, _ = self._vr_bound(x, l, k, 0, True)
            else:
                inputs = x + [l]
                lower_bound, _, _ = self._elbo(x, l, 1, True)
                lower_bound = T.sum(lower_bound, axis=1)
        else:
            inputs = x + [l, k]
            lower_bound = self._vr_bound_test(x, l, k, missing, True)

        self.lower_bound_test = theano.function(inputs=inputs,
                                                outputs=lower_bound,
                                                on_unused_input='ignore')

    def test(self, test_set, l=1, k=1, type_p="normal",
             missing=False, n_batch=None, verbose=True):

        self._set_test(type_p, missing)
        return super(CMMA, self).test(test_set,
                                      l, k, n_batch, verbose)

    def _elbo(self, x, l, annealing_beta, deterministic=False):
        """
        The evidence lower bound (original VAE)
        [Kingma+ 2013] Auto-Encoding Variational Bayes
        """

        z = self.q.sample_given_x(x, repeat=l,
                                  deterministic=deterministic)

        inverse_z = self._inverse_samples(self._select_input(z, [0]), prior_mode=self.prior_mode)
        log_likelihood =\
            self.p[1].log_likelihood_given_x(inverse_z,
                                             deterministic=deterministic)

        kl_qp = analytical_kl(self.q, self.p[0],
                              given=[x, [x[1]]],
                              deterministic=deterministic)

        lower_bound = T.stack([-kl_qp, log_likelihood], axis=-1)
        loss = -T.mean(log_likelihood - annealing_beta * kl_qp)

        q_params = self.q.get_params()
        p_params = self.p[0].get_params() + self.p[1].get_params()
        params = q_params + p_params

        return lower_bound, loss, params

    def _vr_bound(self, x, l, k, iw_alpha=0, deterministic=False):
        q_samples = self.q.sample_given_x(
            x, repeat=l * k, deterministic=deterministic)
        log_iw = self._log_importance_weight(q_samples,
                                             deterministic=deterministic)
        log_iw_matrix = log_iw.reshape((x[0].shape[0] * l, k))
        log_likelihood = log_mean_exp(
            log_iw_matrix, axis=1, keepdims=True)

        log_likelihood = log_likelihood.reshape((x[0].shape[0], l))
        log_likelihood = T.mean(log_likelihood, axis=1)
        loss = -T.mean(log_likelihood)

        p_params = []
        for i, p in enumerate(self.p):
            p_params += self.p[i].get_params()
        q_params = self.q.get_params()
        params = q_params + p_params

        return log_likelihood, loss, params

    def _vr_bound_test(self, x, l, k, missing=False, deterministic=False):
        """
        Paramaters
        ----------
        x : TODO

        l : TODO

        k : TODO

        Returns
        --------
        log_marginal_estimate : array, shape (n_samples)
           Estimated log likelihood.
        """

        n_x = x[0].shape[0]
        rep_x = [T.extra_ops.repeat(_x, l * k, axis=0) for _x in x]

        if missing:
            NotImplementedError

        else:
            samples = self.q.sample_given_x(rep_x, deterministic=True)
            log_iw = self._log_reconstruct_weight(samples, deterministic=True)

        log_iw_matrix = T.reshape(log_iw, (n_x * l, k))
        log_likelihood = log_mean_exp(
            log_iw_matrix, axis=1, keepdims=True)
        log_likelihood = log_likelihood.reshape((x[0].shape[0], l))
        log_likelihood = T.mean(log_likelihood, axis=1)

        return log_likelihood

    def _log_importance_weight(self, samples, deterministic=False):
        """
        inputs : [[x,y],z1,z2,...,zn]
        outputs : log p(x,z1,z2,...,zn|y)/q(z1,z2,...,zn|x,y)
        """
        log_iw = 0

        """
        log q(z1,z2,...,zn|x,y)
        samples : [[x,y],z1,z2,...,zn]
        """
        q_log_likelihood =\
            self.q.log_likelihood_given_x(samples,
                                          deterministic=deterministic)

        """
        log p(z1,z2,...,zn|y)
        inverse_samples : [y,zn,,zn-1,...,x]
        """
        samples_0 = self._select_input(samples, [1])
        p0_log_likelihood =\
            self.p[0].log_likelihood_given_x(samples_0,
                                             deterministic=deterministic)

        """
        log p(x|z1,z2,...,zn)
        inverse_samples : [zn,zn-1,...,x]
        """
        samples_1 = self._select_input(samples, [0])
        p_samples = self._inverse_samples(
            samples_1, prior_mode=self.prior_mode)

        p1_log_likelihood =\
            self.p[1].log_likelihood_given_x(p_samples,
                                             deterministic=deterministic)

        log_iw += p0_log_likelihood + p1_log_likelihood - q_log_likelihood

        return log_iw

    def _log_reconstruct_weight(self, samples, deterministic=False):
        """
        Paramaters
        ----------
        samples : list
           [[x0,x1,...],z1,z2,...,zn]

        Returns
        -------
        log_iw : array, shape (n_samples*k)
           Estimated log likelihood.
           log p(x1|z1,z2,...,zn)
        """

        log_iw = 0
        # log p(x1|z1,...)
        p_samples, prior_samples = self._inverse_samples(
            self._select_input(samples, [0]),
            prior_mode=self.prior_mode, return_prior=True)
        p_log_likelihood = self.p[1].log_likelihood_given_x(
            p_samples, deterministic=deterministic)

        log_iw += p_log_likelihood
        # log p(z1,,z2,...|zn)
        if self.prior_mode == "MultiPrior":
            log_iw += self.prior.log_likelihood_given_x(prior_samples,
                                                        add_prior=False)

        return log_iw

    def _select_input(self, samples, index=[0], set_zeros=False, inputs=None):
        """
        Paramaters
        ----------
        samples : list
           [[x,y,...],z1,z2,....]

        index : list
           Selects an input from [x,y...].

        set_zero :TODO

        inputs : list
           The inputs which you want to replace from [x,y,...].

        Returns
        ----------
        _samples : list
           if i=[0], then _samples = [[x],z1,z2,....]
           if i=[1], then _samples = [[y],z1,z2,....]
           if i=[0,1], then _samples = [[x,y],z1,z2,....]
           if i=[0] and set_zeros=True, then _samples = [[x,0],z1,z2,....]
        """

        _samples = copy(samples)
        if inputs:
            _samples[0] = tolist(inputs)
        else:
            _samples_inputs = copy(_samples[0])
            if set_zeros:
                for i in self._reverse_index(index):
                    _samples_inputs[i] = T.zeros_like(_samples_inputs[i])
                _samples[0] = _samples_inputs
            else:
                _input_samples = []
                for i in index:
                    _input_samples.append(_samples[0][i])
                _samples[0] = _input_samples
        return _samples
