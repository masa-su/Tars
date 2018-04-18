import numpy as np
import theano
import theano.tensor as T
import lasagne
from progressbar import ProgressBar

from ..utils import log_sum_exp
from ..distributions.estimate_kl import analytical_kl, gauss_gauss_kl
from ..models import VAE


class VZSL(VAE):

    def __init__(self, q, p, prior,
                 n_batch=100, optimizer=lasagne.updates.adam,
                 optimizer_params={},
                 clip_grad=None, max_norm_constraint=None,
                 seed=1234):
        super(VZSL, self).__init__(q, p, prior,
                                   n_batch=n_batch,
                                   optimizer=optimizer,
                                   optimizer_params=optimizer_params,
                                   clip_grad=clip_grad,
                                   max_norm_constraint=max_norm_constraint,
                                   train_iw=False,
                                   test_iw=False,
                                   seed=seed)

    def _set_train(self, l, k, annealing_beta):
        # set inputs
        x = self.q.inputs
        a = self.prior.inputs
        A = T.fmatrix("A")
        regularizer_rate = T.fscalar("regularizer_rate")

        # training
        inputs = x + a + [A, l, annealing_beta, regularizer_rate]
        lower_bound, loss, params = self._elbo(x, a, A, l,
                                               annealing_beta,
                                               regularizer_rate,
                                               False)

        lower_bound = T.mean(lower_bound, axis=0)
        updates = self._get_updates(loss, params, self.optimizer,
                                    self.optimizer_params, self.clip_grad,
                                    self.max_norm_constraint)

        self.lower_bound_train = theano.function(inputs=inputs,
                                                 outputs=lower_bound,
                                                 updates=updates,
                                                 on_unused_input='ignore')

    def _set_test(self, l, k):
        # set inputs
        x = self.q.inputs
        a = self.prior.inputs
        A = T.fmatrix("A")

        # test
        inputs = x + a + [A, l]
        lower_bound, _, _ = self._elbo(x, a, A, l, 1, 1, True)
        lower_bound = T.sum(lower_bound, axis=1)

        self.lower_bound_test = theano.function(inputs=inputs,
                                                outputs=lower_bound,
                                                on_unused_input='ignore')

        # test class
        inputs = x + [A]
        regularizer_kl = self._margin_regularizer(x, A,
                                                  True)  # (n_sample, n_class)
        regularizer_kl = T.argmin(regularizer_kl, axis=1)  # (n_sample,)

        self.class_test = theano.function(inputs=inputs,
                                          outputs=regularizer_kl,
                                          on_unused_input='ignore')

    def train(self, train_set, A, l=1,
              annealing_beta=1, regularizer_rate=1,
              verbose=False):
        n_x = train_set[0].shape[0]
        nbatches = n_x // self.n_batch
        lower_bound_all = []

        if verbose:
            pbar = ProgressBar(maxval=nbatches).start()
        for i in range(nbatches):
            start = i * self.n_batch
            end = start + self.n_batch
            batch_x = [_x[start:end] for _x in train_set]

            _x = batch_x + [A, l, annealing_beta, regularizer_rate]
            lower_bound = self.lower_bound_train(*_x)
            lower_bound_all.append(np.array(lower_bound))

            if verbose:
                pbar.update(i)

        lower_bound_all = np.mean(lower_bound_all, axis=0)
        return lower_bound_all

    def test(self, test_set, A, l=1, n_batch=None, verbose=True):
        if n_batch is None:
            n_batch = self.n_batch

        n_x = test_set[0].shape[0]
        nbatches = n_x // n_batch
        lower_bound_all = []

        if verbose:
            pbar = ProgressBar(maxval=nbatches).start()
        for i in range(nbatches):
            start = i * n_batch
            end = start + n_batch
            batch_x = [_x[start:end] for _x in test_set]

            _x = batch_x + [A, l]
            lower_bound = self.lower_bound_test(*_x)
            lower_bound_all = np.r_[lower_bound_all, lower_bound]

            if verbose:
                pbar.update(i)

        return lower_bound_all

    def predict_class(self, test_set, A, n_batch=None, verbose=True):
        if n_batch is None:
            n_batch = self.n_batch

        n_x = test_set[0].shape[0]
        nbatches = n_x // n_batch
        lower_bound_all = []

        if verbose:
            pbar = ProgressBar(maxval=nbatches).start()
        for i in range(nbatches):
            start = i * n_batch
            end = start + n_batch
            batch_x = [_x[start:end] for _x in test_set]

            _x = batch_x + [A]
            lower_bound = self.class_test(*_x)
            lower_bound_all = np.r_[lower_bound_all, lower_bound]

            if verbose:
                pbar.update(i)

        batch_x = [_x[end:] for _x in test_set]
        _x = batch_x + [A]
        lower_bound = self.class_test(*_x)
        lower_bound_all = np.r_[lower_bound_all, lower_bound]

        return lower_bound_all

    def _elbo(self, x, a, A, l,
              annealing_beta, regularizer_rate, deterministic=False):
        """
        Variational zero-shot learning
        """

        kl_divergence = analytical_kl(self.q, self.prior,
                                      given=[x, a],
                                      deterministic=deterministic)
        z = self.q.sample_given_x(x, repeat=l,
                                  deterministic=deterministic)

        inverse_z = self._inverse_samples(z)
        log_likelihood =\
            self.p.log_likelihood_given_x(inverse_z,
                                          deterministic=deterministic)

        regularizer_kl =\
            self._margin_regularizer(x, A,
                                     deterministic=deterministic)  # (n_sample, n_class)
        regularizer_kl = log_sum_exp(-regularizer_kl, axis=1)  # (n_sample,)

        lower_bound = T.stack([-kl_divergence,
                               log_likelihood, -regularizer_kl], axis=-1)
        loss = -T.mean(log_likelihood
                       - annealing_beta * kl_divergence
                       - regularizer_rate * regularizer_kl)

        q_params = self.q.get_params()
        p_params = self.p.get_params()
        prior_params = self.prior.get_params()
        params = q_params + p_params + prior_params

        return lower_bound, loss, params

    def _margin_regularizer(self, x, A, deterministic=False):
        """
        x: list, samples [(n_sample, x_dim)]
        A: class-attribute matrix (n_class, a_dim)
        """

        z_x_mean, z_x_var =\
            self.q.fprop(x,
                         deterministic=deterministic)  # (n_sample, z_dim)
        z_a_mean, z_a_var =\
            self.prior.fprop([A],
                             deterministic=deterministic)  # (n_class, z_dim)

        # (n_sample, z_dim) -> (n_sample, z_dim, newaxis)
        z_x_mean = z_x_mean[:, :, np.newaxis]
        z_x_var = z_x_var[:, :, np.newaxis]

        # (n_class, z_dim) -> (newaxis, z_dim, n_class)
        z_a_mean = z_a_mean.T[np.newaxis, :, :]
        z_a_var = z_a_var.T[np.newaxis, :, :]

        regularizer_kl =\
            gauss_gauss_kl(z_x_mean, z_x_var,
                           z_a_mean, z_a_var)  # (n_sample, n_class)

        return regularizer_kl
