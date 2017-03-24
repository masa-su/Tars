from copy import copy, deepcopy

import theano
import theano.tensor as T
import lasagne
import numpy as np
from progressbar import ProgressBar

from ..utils import tolist, log_mean_exp
from . import VAE


class SS_JMVAE(VAE):

    def __init__(self, q, p, f, s_q, prior=None,
                 n_batch=100, n_batch_u=100,
                 optimizer=lasagne.updates.adam,
                 optimizer_params={},
                 clip_grad=None, max_norm_constraint=None,
                 seed=1234):
        self.s_q = s_q
        self.f = f
        self.n_batch_u = n_batch_u

        super(SS_JMVAE,
              self).__init__(q, p, prior=prior,
                             n_batch=n_batch,
                             optimizer=optimizer,
                             optimizer_params=optimizer_params,
                             clip_grad=clip_grad,
                             max_norm_constraint=max_norm_constraint,
                             train_iw=True, test_iw=True,
                             iw_alpha=0, seed=seed)

        # inputs
        x_u = self.q.inputs
        x_l = deepcopy(self.q.inputs)
        y = T.fmatrix("y")
        l = T.iscalar("l")
        k = T.iscalar("k")

        # training
        rate = T.fscalar("rate")
        inputs = x_u + x_l + [y, l, k, rate]
        lower_bound_u, loss_u, params = self._vr_bound(x_u, l, k, 0, False)
        lower_bound_l, loss_l, _ = self._vr_bound(x_l, l, k, 0, False,
                                                  tolist(y))
        lower_bound_y, loss_y, _ = self._discriminate(x_l, tolist(y), False)

        lower_bound = [T.mean(lower_bound_u), T.mean(lower_bound_l),
                       T.mean(lower_bound_y)]

        loss = loss_u + loss_l + rate*loss_y
        updates = self._get_updates(loss, params, optimizer, optimizer_params,
                                    clip_grad, max_norm_constraint)

        self.lower_bound_train = theano.function(inputs=inputs,
                                                 outputs=lower_bound,
                                                 updates=updates,
                                                 on_unused_input='ignore')

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

    def train(self, train_set_u, train_set_l, l=1, k=1,
              nbatches=2000, get_batch_samples=None,
              discriminate_rate=1, verbose=False):
        lower_bound_all = []

        if verbose:
            pbar = ProgressBar(maxval=nbatches).start()

        for i in range(nbatches):
            # unlabel
            batch_set_u = get_batch_samples(train_set_u,
                                            n_batch=self.n_batch_u)
            # label
            batch_set_l = get_batch_samples(train_set_l,
                                            n_batch=self.n_batch)

            _x = batch_set_u + batch_set_l + [l, k, discriminate_rate]
            lower_bound = self.lower_bound_train(*_x)

            lower_bound_all.append(np.array(lower_bound))

            if verbose:
                pbar.update(i)

        lower_bound_all = np.mean(lower_bound_all, axis=0)
        return lower_bound_all

    def test(self, test_set_u, test_set_l, l=1, k=1,
             n_batch=None, verbose=True):
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

            classifier = self.classifier_test(*batch_x_l)
            lower_bound.append(classifier)

            lower_bound_all.append(np.array(lower_bound))

            if verbose:
                pbar.update(i)

        return lower_bound_all

    def _discriminate(self, x, y, deterministic=False):
        # log q(y|x0,...xn)
        # _samples : [[x0,...,xn],y]
        _samples = [x] + y
        log_likelihood = self.f.log_likelihood_given_x(
            _samples, deterministic=deterministic)
        loss = -T.mean(log_likelihood)
        params = self.f.get_params()

        return log_likelihood, loss, params

    def _elbo(self, x, l, annealing_beta, deterministic=False):
        raise NotImplementedError

    def _vr_bound(self, x, l=1, k=1,
                  iw_alpha=0, deterministic=False, y=None):
        if y is None:
            supervised = False
        else:
            supervised = True
        rep_x = [T.extra_ops.repeat(_x, l * k, axis=0) for _x in x]

        if supervised is False:
            # y ~ q(y|x)
            rep_y = self.f.sample_given_x(rep_x,
                                          deterministic=deterministic)[-1:]
        else:
            rep_y = [T.extra_ops.repeat(_y, l * k,
                                        axis=0) for _y in y]

        # x,a1,..,aL ~ q(a|x)
        a = self.q.sample_given_x(rep_x, deterministic=deterministic)
        # z ~ q(z|a,y)
        z = self.s_q.sample_given_x(a[-1:]+rep_y,
                                    deterministic=deterministic)[-1:]
        # q_samples = [a1,...,aL,z]
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

        # log q(z|aL,y)
        # _samples : [[aL,y],z]
        _samples = [tolist(samples[-2])+y, samples[-1]]
        q_log_likelihood += self.s_q.log_likelihood_given_x(
            _samples, deterministic=deterministic)

        if supervised is False:
            # log q(y|x0,...xn)
            # _samples : [[x0,...,xn],y]
            _samples = [samples[0]]+y
            q_log_likelihood += self.f.log_likelihood_given_x(
                _samples, deterministic=deterministic)

        # log p(x0,...,xn|a1)
        # samples : [[x1,...,xn],a1...,aL,z]
        p_log_likelihood_all = []
        for i, p in enumerate(self.p):
            p_samples, _ = self._inverse_samples(
                self._select_input(samples, [i]), return_prior=True)
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
        log_iw += 1 / 2.  # Bernoulli prior distribution

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
