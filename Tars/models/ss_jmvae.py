from copy import copy

import theano
import theano.tensor as T
import lasagne

from ..utils import tolist, log_mean_exp
from ..distributions.estimate_kl import analytical_kl
from . import VAE


class SS_JMVAE(VAE):

    def __init__(self, q, p, f, s_q, prior=None,
                 n_batch=100, optimizer=lasagne.updates.adam,
                 optimizer_params={},
                 clip_grad=None, max_norm_constraint=None,
                 train_iw=True, test_iw=True, seed=1234):
        self.s_q = s_q
        self.f = f
        super(SS_JMVAE,
              self).__init__(q, p, prior=prior,
                             n_batch=n_batch,
                             optimizer=optimizer,
                             optimizer_params=optimizer_params,
                             clip_grad=clip_grad,
                             max_norm_constraint=max_norm_constraint,
                             train_iw=train_iw, test_iw=test_iw,
                             iw_alpha=0, seed=seed)

    def _elbo(self, x, l, annealing_beta, deterministic=False):
        raise NotImplementedError

    def _vr_bound(self, x, l=1, k=1,
                  iw_alpha=0, deterministic=False, y=None):
        if y is None:
            # y ~ q(y|x)
            y = self.f.sample_given_x(x, repeat=l * k,
                                      deterministic=deterministic)[-1:]

        # x,z1,..,zL ~ q(z|x)
        z = self.q.sample_given_x(x, repeat=l * k,
                                  deterministic=deterministic)
        # s ~ q(s|zL,y)
        s = self.s_q.sample_given_x(z[-1:]+y, repeat=l * k,
                                    deterministic=deterministic)[-1:]
        # q_samples = [z1,...,zL,s]
        q_samples = z+s
        if y is None:
            supervised = False
        else:
            supervised = True
        log_iw = self._log_importance_weight(q_samples, y,
                                             supervised=supervised
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
        s_q_params = self.s_q.get_params()
        f_params = self.f.get_params()
        params = f_params + s_q_params + p_params

        if self.prior_mode == "MultiPrior":
            params += self.prior.get_params()

        return log_likelihood, loss, params

    def _log_importance_weight(self, samples, y, supervised=True,
                               deterministic=True):
        """
        Paramaters
        ----------
        samples : list
           [[x0,..,xn],z1,z2,...,zL,s]

        samples : y
           [y]

        Returns
        -------
        log_iw : array, shape (n_samples)
           Estimated log likelihood.
           log p(x0,...xn,z1,z2,...,zL,y)/q(z1,z2,...,zL,y|x0,...xn)
        """

        # samples : [[x0,..,xn],z1,z2,...,zL,s]
        log_iw = 0

        # log q(z1,z2,...,zL|x0,...xn)
        # _samples : [[x0,..,xn],z1,z2,...,zL]
        _samples = samples[:-1]
        q_log_likelihood = self.q.log_likelihood_given_x(
            _samples, deterministic=deterministic)

        # log q(s|zL,y)
        # _samples : [[zL,y],s]
        _samples = [tolist(samples[-2])+y, samples[-1]]
        q_log_likelihood += self.s_q.log_likelihood_given_x(
            _samples, deterministic=deterministic)

        if supervised is False:
            # log q(y|x0,...xn)
            # _samples : [[zL],y]
            _samples = [tolist(samples[-2])]+y
            q_log_likelihood += self.f.log_likelihood_given_x(
                _samples, deterministic=deterministic,
                last_layer=True)

        # log p(x0,...,xn|z1)
        # samples : [[x1,...,xn],z1...,zL,s]
        p_log_likelihood_all = []
        for i, p in enumerate(self.p):
            p_samples, _ = self._inverse_samples(
                self._select_input(samples, [i]), return_prior=True)
            p_log_likelihood = self.p[i].log_likelihood_given_x(
                p_samples, deterministic=deterministic)
            p_log_likelihood_all.append(p_log_likelihood)
        log_iw += sum(p_log_likelihood_all) - q_log_likelihood

        # log p(z1,..,zL|s,y)
        # prior_samples : [[s],zL,...,z1]
        _, prior_samples = self._inverse_samples(
            self._select_input(samples, [0]), return_prior=True)
        
        prior_samples = self._select_input(prior_samples,
                                           inputs=prior_samples[0]+y)

        if self.prior_mode == "MultiPrior":
            log_iw += self.prior.log_likelihood_given_x(prior_samples)
        else:
            log_iw += self.prior.log_likelihood(prior_samples)

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

    def _reverse_index(self, index=[0]):
        N = len(self.p)
        return list(set(range(N)) - set(index))
