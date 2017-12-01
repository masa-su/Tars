from copy import copy, deepcopy

import theano
import theano.tensor as T
import lasagne
import numpy as np
from progressbar import ProgressBar

from ..utils import tolist, log_mean_exp
from . import VAE


class SS_VAE(VAE):

    def __init__(self, q, p, f, prior=None, c=None,
                 n_batch=100, n_batch_u=100,
                 optimizer=lasagne.updates.adam,
                 optimizer_params={},
                 sum_classes=False,
                 clip_grad=None, max_norm_constraint=None,
                 regularization_penalty=None,
                 seed=1234):
        self.f = f
        self.c = c
        self.n_batch_u = n_batch_u
        self.regularization_penalty = regularization_penalty
        self.sum_classes=sum_classes

        super(SS_VAE,
              self).__init__(q, p, prior=prior,
                             n_batch=n_batch,
                             optimizer=optimizer,
                             optimizer_params=optimizer_params,
                             clip_grad=clip_grad,
                             max_norm_constraint=max_norm_constraint,
                             train_iw=True, test_iw=True,
                             iw_alpha=0, seed=seed)


    def _set_train(self, l, k, *args):
        # inputs
        x_u = self.q.inputs[:-1]
        x_l = deepcopy(self.q.inputs[:-1])
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
        if self.regularization_penalty:
            loss += self.regularization_penalty
        updates = self._get_updates(loss, params, self.optimizer,
                                    self.optimizer_params, self.clip_grad,
                                    self.max_norm_constraint)

        self.lower_bound_train = theano.function(inputs=inputs,
                                                 outputs=lower_bound,
                                                 updates=updates,
                                                 on_unused_input='ignore')

        """
        # training (f params)
        params = self.f.get_params()
        updates = self._get_updates(loss, params, self.optimizer,
                                    self.optimizer_params, self.clip_grad,
                                    self.max_norm_constraint)

        self.lower_bound_train_f = theano.function(inputs=inputs,
                                                   outputs=lower_bound,
                                                   updates=updates,
                                                   on_unused_input='ignore')

        # training (without f params)
        _, _, params = self._vr_bound(x_l, l, k, 0, False,
                                      tolist(y))
        updates = self._get_updates(loss, params, self.optimizer,
                                    self.optimizer_params, self.clip_grad,
                                    self.max_norm_constraint)

        self.lower_bound_train_w_f = theano.function(inputs=inputs,
                                                     outputs=lower_bound,
                                                     updates=updates,
                                                     on_unused_input='ignore')
        """
        # training (non classifier)
        loss = loss_u + loss_l
        if self.regularization_penalty:
            loss += self.regularization_penalty
        updates = self._get_updates(loss, params, self.optimizer,
                                    self.optimizer_params, self.clip_grad,
                                    self.max_norm_constraint)

        self.lower_bound_non_classifier = theano.function(inputs=inputs,
                                                          outputs=lower_bound,
                                                          updates=updates,
                                                          on_unused_input='ignore')

        # training (lower bound classification)
        _, loss, params = self._discriminate(x_l, tolist(y), False)
        loss = rate * loss
        if self.regularization_penalty:
            loss += self.regularization_penalty
        updates = self._get_updates(loss, params, self.optimizer,
                                    self.optimizer_params, self.clip_grad,
                                    self.max_norm_constraint)

        self.lower_bound_classifier = theano.function(inputs=inputs,
                                                      outputs=lower_bound,
                                                      updates=updates,
                                                      on_unused_input='ignore')

        # training (classification)
        inputs = x_l + [y]
        lower_bound_y, loss, params = self._discriminate(x_l, tolist(y), False)
        if self.regularization_penalty:
            loss += self.regularization_penalty
        updates = self._get_updates(loss, params, self.optimizer,
                                    self.optimizer_params, self.clip_grad,
                                    self.max_norm_constraint)

        self.classifier_train = theano.function(inputs=inputs,
                                                outputs=T.mean(lower_bound_y),
                                                updates=updates,
                                                on_unused_input='ignore')

    def _set_test(self, l, k):
        # inputs
        x_u = self.q.inputs[:-1]
        x_l = deepcopy(self.q.inputs[:-1])
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

    def train(self, train_set_u, train_set_l, l=1, k=1,
              nbatches=2000, get_batch_samples=None,
              discriminate_rate=1, non_classifier=False,
              verbose=False, **kwargs):
        lower_bound_all = []

        if verbose:
            pbar = ProgressBar(maxval=nbatches).start()

        _n_batch_u = len(train_set_u[0]) / nbatches
        _n_batch = len(train_set_l[0]) / nbatches

        for i in range(nbatches):
            if get_batch_samples:
                # unlabel
                batch_set_u = get_batch_samples(train_set_u,
                                                n_batch=self.n_batch_u)
                # label
                batch_set_l = get_batch_samples(train_set_l,
                                                n_batch=self.n_batch)
            else:
                # unlabel
                start_u = i * _n_batch_u
                end_u = start_u + _n_batch_u
                batch_set_u = [_x[start_u:end_u] for _x in train_set_u]

                # label
                start = i * _n_batch
                end = start + _n_batch
                batch_set_l = [_x[start:end] for _x in train_set_l]
            """
            _x = batch_set_u + batch_set_l + [l, k, discriminate_rate]
            lower_bound = self.lower_bound_train_f(*_x)
            lower_bound = self.lower_bound_train_w_f(*_x)
            """
            _x = batch_set_u + batch_set_l + [l, k, discriminate_rate]
            if non_classifier:
                lower_bound = self.lower_bound_non_classifier(*_x)
            else:
                lower_bound = self.lower_bound_train(*_x)

            lower_bound_all.append(np.array(lower_bound))

            if verbose:
                pbar.update(i)

        lower_bound_all = np.mean(lower_bound_all, axis=0)
        return lower_bound_all

    def train_classifier(self, train_set_l, nbatches=2000,
                         get_batch_samples=None, verbose=False,
                         **kwargs):

        lower_bound_all = []

        if verbose:
            pbar = ProgressBar(maxval=nbatches).start()

        _n_batch = len(train_set_l[0]) / nbatches

        for i in range(nbatches):
            if get_batch_samples:
                # label
                batch_set_l = get_batch_samples(train_set_l,
                                                n_batch=self.n_batch)
            else:
                # label
                start = i * _n_batch
                end = start + _n_batch
                batch_set_l = [_x[start:end] for _x in train_set_l]

            lower_bound = self.classifier_train(*batch_set_l)
            lower_bound_all.append(np.array(lower_bound))

            if verbose:
                pbar.update(i)

        lower_bound_all = np.mean(lower_bound_all, axis=0)
        return lower_bound_all

    def test(self, test_set_u, test_set_l, l=1, k=1,
             n_batch=None, verbose=True, **kwargs):
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

    def test_classifier(self, test_set_l, n_batch=None, verbose=True,
                        **kwargs):
        if n_batch is None:
            n_batch = self.n_batch

        n_x = test_set_l[0].shape[0]
        nbatches = n_x // n_batch
        lower_bound_all = []

        if verbose:
            pbar = ProgressBar(maxval=nbatches).start()
        for i in range(nbatches):
            # label
            start = i * self.n_batch
            end = start + self.n_batch
            batch_x_l = [_x[start:end] for _x in test_set_l]

            classifier = self.classifier_test(*batch_x_l)
            lower_bound_all.append(np.array(classifier))

            if verbose:
                pbar.update(i)

        return lower_bound_all

    def _discriminate(self, x, y, deterministic=False):
        # log q(y|x)
        # _samples : [[x],y]
        _samples = [x] + y
        log_likelihood = self.f.log_likelihood_given_x(
            _samples, deterministic=deterministic)
        loss = -T.sum(log_likelihood) # mean or sum
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
            if self.sum_classes is True:
                y_dim = rep_y[0].shape[1]
                len_x = rep_x[0].shape[0]
                
                rep_x = [T.extra_ops.repeat(_x, y_dim, axis=0) for _x in rep_x]

                # suppose len(rep_y)==1
                _y = T.tile(T.eye(y_dim), len_x).T
                rep_y = [_y]
        else:
            rep_y = [T.extra_ops.repeat(_y, l * k,
                                        axis=0) for _y in y]

        # z ~ q(z|x,y)
        z = self.q.sample_given_x(rep_x + rep_y,
                                  deterministic=deterministic)[-1:]
        # q_samples = [[x],z]
        q_samples = [rep_x] + z

        # importance weighted
        log_iw = self._log_importance_weight(q_samples, rep_y,
                                             supervised=supervised,
                                             deterministic=deterministic)

        if (self.sum_classes is True) and (supervised is False):
            logp_y_x = self.c.log_likelihood_given_x([rep_x]+rep_y,
                                                     deterministic=deterministic)
            log_iw = log_iw * T.exp(logp_y_x)
            log_iw_matrix = log_iw.reshape((x[0].shape[0] * l * k, y_dim))
            log_iw = T.sum(log_iw_matrix, axis=1)

        log_iw_matrix = log_iw.reshape((x[0].shape[0] * l, k))
        log_likelihood = log_mean_exp(
            log_iw_matrix, axis=1, keepdims=True)

        log_likelihood = log_likelihood.reshape((x[0].shape[0], l))
        log_likelihood = T.mean(log_likelihood, axis=1)
        loss = -T.sum(log_likelihood) # mean or sum

        p_params = []
        p_params = self.p.get_params()
        q_params = self.q.get_params()
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
           [[x],z]

        samples : y
           [y]

        Returns
        -------
        log_iw : array, shape (n_samples)
           Estimated log likelihood.
           supervised=True:
             log p(x,y,z)/q(y,z|x)
           supervised=False:
             log p(x,y,z)/q(z|x,y)
        """

        # samples : [[x],z]
        log_iw = 0

        # log q(z|x,y)
        # samples_y : [[x,y],z]
        samples_y = [tolist(samples[0]) + y, samples[-1]]
        q_log_likelihood = self.q.log_likelihood_given_x(
            samples_y, deterministic=deterministic)

        if supervised is False:
            # log q(y|x)
            # _samples : [[x],y]
            _samples = [tolist(samples[0])] + y
            if self.sum_classes is True:
                q_log_likelihood += self.c.log_likelihood_given_x(
                    _samples, deterministic=deterministic)
            else:
                q_log_likelihood += self.f.log_likelihood_given_x(
                    _samples, deterministic=deterministic)

        # log p(x|z,y)
        # samples : [[z,y],x]
        p_samples, prior_samples = self._inverse_samples(samples_y,
                                                         return_prior=True)
        p_log_likelihood = self.p.log_likelihood_given_x(
            p_samples, deterministic=deterministic)
        log_iw += p_log_likelihood - q_log_likelihood

        if self.prior_mode == "MultiPrior":
            log_iw += self.prior.log_likelihood_given_x(prior_samples)
        else:
            log_iw += self.prior.log_likelihood(prior_samples)
#        log_iw += 1. / np.float32(10)  # Categorical prior distribution

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
