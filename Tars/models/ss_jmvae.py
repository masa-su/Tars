from copy import copy, deepcopy

import theano
import theano.tensor as T
import lasagne
import numpy as np
from progressbar import ProgressBar

from ..utils import tolist, log_mean_exp
from ..distributions.estimate_kl import analytical_kl
from . import VAE


class SS_JMVAE(VAE):

    def __init__(self, q, p, f, s_q, prior=None,
                 n_batch=100, n_batch_u=10000,
                 discriminate_rate=0.1,
                 optimizer=lasagne.updates.adam,
                 optimizer_params={},
                 clip_grad=None, max_norm_constraint=None,
                 seed=1234):
        self.s_q = s_q
        self.f = f
        self.n_batch_u = n_batch_u
        self.discriminate_rate = discriminate_rate

        super(SS_JMVAE,
              self).__init__(q, p, prior=prior,
                             n_batch=n_batch,
                             optimizer=optimizer,
                             optimizer_params=optimizer_params,
                             clip_grad=clip_grad,
                             max_norm_constraint=max_norm_constraint,
                             train_iw=True, test_iw=True,
                             iw_alpha=0, seed=seed)

        # set inputs
        x_u = self.q.inputs
        x_l = deepcopy(self.q.inputs)
        y = T.fmatrix("y")
        l = T.iscalar("l")
        k = T.iscalar("k")

        # training
        if self.train_iw:
            inputs = x_u + x_l + [y, l, k]
            lower_bound_u, loss_u, params = self._vr_bound(x_u, l, k, 0, False)
            lower_bound_l, loss_l, _ = self._vr_bound(x_l, l, k, 0, False, tolist(y))
            lower_bound_y, loss_y = self._discriminate(x_l, tolist(y), False)
        else:
            raise NotImplementedError

        rate = self.discriminate_rate * (self.n_batch + self.n_batch_u) / self.n_batch
        loss = loss_u + loss_l + rate * loss_y

        lower_bound = T.stack([lower_bound_u, lower_bound_l, lower_bound_y], axis=-1)
        lower_bound = T.mean(lower_bound, axis=0)

        updates = self._get_updates(loss, params, optimizer, optimizer_params,
                                    clip_grad, max_norm_constraint)

        self.lower_bound_train = theano.function(inputs=inputs,
                                                 outputs=lower_bound,
                                                 updates=updates,
                                                 on_unused_input='ignore')

        # test
        if self.test_iw:
            inputs = x_u + x_l + [y, l, k]
            lower_bound_u, loss_u, params = self._vr_bound(x_u, l, k, 0, True)
            lower_bound_l, loss_l, _ = self._vr_bound(x_l, l, k, 0, True, tolist(y))
            lower_bound_y, loss_y = self._discriminate(x_l, tolist(y), True)
        else:
            raise NotImplementedError

        lower_bound = T.stack([lower_bound_u, lower_bound_l, lower_bound_y], axis=-1)
        lower_bound = T.mean(lower_bound, axis=0)

        self.lower_bound_test = theano.function(inputs=inputs,
                                                outputs=lower_bound,
                                                on_unused_input='ignore')

    def train(self, train_set_u, train_set_l, l=1, k=1, verbose=False):
        n_x = train_set_l[0].shape[0]
        nbatches = n_x // self.n_batch
        lower_bound_all = []

        if verbose:
            pbar = ProgressBar(maxval=nbatches).start()

        for i in range(nbatches):
            # unlabel
            start = i * self.n_batch_u
            end = start + self.n_batch_u
            batch_x_u = [_x[start:end] for _x in train_set_u]

            # label
            start = i * self.n_batch
            end = start + self.n_batch
            batch_x_l = [_x[start:end] for _x in train_set_l]

            _x = batch_x_u + batch_x_l + [l, k]
            lower_bound = self.lower_bound_train(*_x)
            lower_bound_all.append(np.array(lower_bound))

            if verbose:
                pbar.update(i)

        lower_bound_all = np.mean(lower_bound_all, axis=0)
        return lower_bound_all

    def test(self, test_set_u, test_set_l, l=1, k=1, n_batch=None, verbose=True):
        if n_batch is None:
            n_batch = self.n_batch

        n_x = test_set_l[0].shape[0]
        nbatches = n_x // n_batch
        lower_bound_all = []

        if verbose:
            pbar = ProgressBar(maxval=nbatches).start()
        for i in range(nbatches):
            # unlabel
            start = i * self.n_batch_u
            end = start + self.n_batch_u
            batch_x_u = [_x[start:end] for _x in test_set_u]

            # label
            start = i * self.n_batch
            end = start + self.n_batch
            batch_x_l = [_x[start:end] for _x in test_set_l]

            _x = batch_x_u + batch_x_l + [l, k]
            lower_bound = self.lower_bound_test(*_x)
            lower_bound_all.append(np.array(lower_bound))

            if verbose:
                pbar.update(i)

        return lower_bound_all

    def _discriminate(self, x, y, deterministic=False):
        # log q(y|x0,...xn)
        # _samples : [[x0,...,xn],y]
        _samples = [x]+y
        log_likelihood = self.f.log_likelihood_given_x(
            _samples, deterministic=deterministic,
            last_layer=True)
        loss = -T.mean(log_likelihood)

        return log_likelihood, loss

    def _elbo(self, x, l, annealing_beta, deterministic=False):
        raise NotImplementedError

    def _vr_bound(self, x, l=1, k=1,
                  iw_alpha=0, deterministic=False, y=None):
        if y is None:
            supervised = False
        else:
            supervised = True

        if supervised is False:
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

        # importance weighted
        log_iw = self._log_importance_weight(q_samples, y,
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
            p_params += self.p[i].get_params()
        s_q_params = self.s_q.get_params()
        params = s_q_params + p_params

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
            # _samples : [[x0,...,xn],y]
            _samples = [samples[0]]+y
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
