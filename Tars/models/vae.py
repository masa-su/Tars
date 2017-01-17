import numpy as np
import theano
import theano.tensor as T
import lasagne
from progressbar import ProgressBar

from ..utils import log_mean_exp
from ..distributions.estimate_kl import analytical_kl, get_prior
from ..models.model import Model


class VAE(Model):

    def __init__(self, q, p, prior=None,
                 n_batch=100, optimizer=lasagne.updates.adam,
                 optimizer_params={},
                 clip_grad=None, max_norm_constraint=None,
                 train_iw=False, test_iw=True, iw_alpha=0,
                 seed=1234):
        super(VAE, self).__init__(n_batch=n_batch, seed=seed)

        self.q = q
        self.p = p
        if prior:
            self.prior = prior
        else:
            self.prior = get_prior(self.q)
        self.train_iw = train_iw
        self.test_iw = test_iw

        # set inputs
        x = self.q.inputs
        l = T.iscalar("l")
        k = T.iscalar("k")
        annealing_beta = T.fscalar("beta")

        # training
        if self.train_iw:
            inputs = x + [l, k]
            lower_bound, loss, params = self._vr_bound(x, l, k,
                                                       iw_alpha, False)
        else:
            inputs = x + [l, annealing_beta]
            lower_bound, loss, params = self._elbo(x, l, annealing_beta, False)
        lower_bound = T.mean(lower_bound, axis=0)
        updates = self._get_updates(loss, params, optimizer, optimizer_params,
                                    clip_grad, max_norm_constraint)

        self.lower_bound_train = theano.function(inputs=inputs,
                                                 outputs=lower_bound,
                                                 updates=updates,
                                                 on_unused_input='ignore')

        # test
        if self.test_iw:
            inputs = x + [l, k]
            lower_bound, _, _ = self._vr_bound(x, l, k, 0, True)
        else:
            inputs = x + [l]
            lower_bound, _, _ = self._elbo(x, l, 1, True)
            lower_bound = T.sum(lower_bound, axis=1)

        self.lower_bound_test = theano.function(inputs=inputs,
                                                outputs=lower_bound,
                                                on_unused_input='ignore')

    def train(self, train_set, l=1, k=1, annealing_beta=1,
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

            if self.train_iw:
                _x = batch_x + [l, k]
                lower_bound = self.lower_bound_train(*_x)
            else:
                _x = batch_x + [l, annealing_beta]
                lower_bound = self.lower_bound_train(*_x)
            lower_bound_all.append(np.array(lower_bound))

            if verbose:
                pbar.update(i)

        lower_bound_all = np.mean(lower_bound_all, axis=0)
        return lower_bound_all

    def test(self, test_set, l=1, k=1, n_batch=None, verbose=True):
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

            if self.test_iw:
                _x = batch_x + [l, k]
                lower_bound = self.lower_bound_test(*_x)
            else:
                _x = batch_x + [l]
                lower_bound = self.lower_bound_test(*_x)
            lower_bound_all = np.r_[lower_bound_all, lower_bound]

            if verbose:
                pbar.update(i)

        return lower_bound_all

    def _elbo(self, x, l, annealing_beta, deterministic=False):
        """
        The evidence lower bound (original VAE)
        [Kingma+ 2013] Auto-Encoding Variational Bayes
        """

        kl_divergence = analytical_kl(self.q, self.prior,
                                      given=[x, None],
                                      deterministic=deterministic)
        z = self.q.sample_given_x(x, repeat=l,
                                  deterministic=deterministic)
        inverse_z = self._inverse_samples(z)
        log_likelihood =\
            self.p.log_likelihood_given_x(inverse_z,
                                          deterministic=deterministic)

        lower_bound = T.stack([-kl_divergence, log_likelihood], axis=-1)
        loss = -T.mean(log_likelihood - annealing_beta * kl_divergence)

        q_params = self.q.get_params()
        p_params = self.p.get_params()
        params = q_params + p_params

        return lower_bound, loss, params

    def _vr_bound(self, x, l, k, iw_alpha=0, deterministic=False):
        """
        Variational Renyi bound
        [Li+ 2016] Renyi Divergence Variational Inference
        [Burda+ 2015] Importance Weighted Autoencoders
        """
        q_samples = self.q.sample_given_x(
            x, repeat=l * k, deterministic=deterministic)
        log_iw = self._log_importance_weight(q_samples,
                                             deterministic=deterministic)
        log_iw_matrix = log_iw.reshape((x[0].shape[0] * l, k))

        if iw_alpha == 1:
            log_likelihood = T.mean(
                log_iw_matrix, axis=1)

        elif iw_alpha == -np.inf:
            log_likelihood = T.max(
                log_iw_matrix, axis=1)

        else:
            log_iw_matrix = log_iw_matrix * (1 - iw_alpha)
            log_likelihood = log_mean_exp(
                log_iw_matrix, axis=1, keepdims=True) / (1 - iw_alpha)

        log_likelihood = log_likelihood.reshape((x[0].shape[0], l))
        log_likelihood = T.mean(log_likelihood, axis=1)
        loss = -T.mean(log_likelihood)

        q_params = self.q.get_params()
        p_params = self.p.get_params()
        params = q_params + p_params

        return log_likelihood, loss, params

    def _log_importance_weight(self, samples, deterministic=False):
        """
        inputs : [[x,y,...],z1,z2,...,zn]
        outputs : log p(x,z1,z2,...,zn|y,...)/q(z1,z2,...,zn|x,y,...)
        """
        log_iw = 0

        """
        log q(z1,z2,...,zn|x,y,...)
        samples : [[x,y,...],z1,z2,...,zn]
        """
        q_log_likelihood =\
            self.q.log_likelihood_given_x(samples,
                                          deterministic=deterministic)

        """
        log p(x|z1,z2,...,zn,y,...)
        inverse_samples : [[zn,y,...],zn-1,...,x]
        """
        inverse_samples = self._inverse_samples(samples)
        p_log_likelihood =\
            self.p.log_likelihood_given_x(inverse_samples,
                                          deterministic=deterministic)

        log_iw += p_log_likelihood - q_log_likelihood
        log_iw += self.prior.log_likelihood(samples[-1])

        return log_iw
