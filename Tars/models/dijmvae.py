from copy import copy, deepcopy

import theano
import theano.tensor as T
import lasagne

from ..utils import tolist, log_mean_exp
from ..distributions.estimate_kl import analytical_kl
from . import JMVAE_KL


class DIJMVAE(JMVAE_KL):

    def __init__(self, q, p, pseudo_q, f, prior=None,
                 gamma=1, sl_rate=1, train_source=True,
                 n_batch=100, optimizer=lasagne.updates.adam,
                 optimizer_params={}, annealing_beta=1,
                 clip_grad=None, max_norm_constraint=None,
                 test_iw=True, seed=1234):
        self.f = f
        self.annealing_beta = annealing_beta
        self.sl_rate = sl_rate
        self.train_source = train_source
        super(DIJMVAE,
              self).__init__(q, p, pseudo_q,
                             prior=prior, gamma=gamma,
                             n_batch=n_batch,
                             optimizer=optimizer,
                             optimizer_params=optimizer_params,
                             clip_grad=clip_grad,
                             max_norm_constraint=max_norm_constraint,
                             test_iw=True, #TODO: fix later
                             seed=seed)

    def _set_train(self, l, k, annealing_beta):
        # set inputs
        x = self.q.inputs
        if self.train_source:
            x_l = deepcopy(tolist(self.q.inputs[0]))
        else:
            x_l = deepcopy(self.q.inputs)
        y = T.fmatrix("y")

        # training
        inputs = x + x_l + [y, l, annealing_beta]
        lower_bound, loss, params = self._elbo(x, x_l, tolist(y), l, annealing_beta, False)
        lower_bound = T.mean(lower_bound, axis=0)
        updates = self._get_updates(loss, params, self.optimizer,
                                    self.optimizer_params, self.clip_grad,
                                    self.max_norm_constraint)

        self.lower_bound_train = theano.function(inputs=inputs,
                                                 outputs=lower_bound,
                                                 updates=updates,
                                                 on_unused_input='ignore')

    def _set_test(self, l, k, type_p="joint", missing=False,
                  index=[0], sampling_n=1, missing_resample=False):
        # set inputs
        x = self.q.inputs

        if self.test_iw:
            inputs = x + [l, k]
            lower_bound, _, _ = self._vr_bound(x, l, k, 0, True)

        self.lower_bound_test = theano.function(inputs=inputs,
                                                outputs=lower_bound,
                                                on_unused_input='ignore')


    def _supervised_learning(self, x_l, y, l, deterministic=False):
        # supervised learning
        rep_x = [T.extra_ops.repeat(_x, l, axis=0) for _x in x_l]

        if self.train_source:
            # [z] ~ q(z|x1) (source)
            z = self.pseudo_q[0].sample_given_x(rep_x, deterministic=deterministic)[-1:]
        else:
            # [z] ~ q(z|x)
            z = self.q.sample_given_x(rep_x, deterministic=deterministic)[-1:]

        rep_y = [T.extra_ops.repeat(_y, l,
                                    axis=0) for _y in y]

        # log q(y|z)
        # _samples : [[z], y]
        _samples = [z] + rep_y
        log_likelihood = self.f.log_likelihood_given_x(
            _samples, deterministic=deterministic)

        log_likelihood = log_likelihood.reshape((rep_x[0].shape[0], l))
        log_likelihood = T.mean(log_likelihood, axis=1, keepdims=True)
        loss = -T.mean(log_likelihood)
        params = self.f.get_params()

        return log_likelihood, loss, params

    def _elbo(self, x, x_l, y, l, annealing_beta, deterministic=False):
        lower_bound, loss, params = \
            super(DIJMVAE, self)._elbo(x, l, annealing_beta,
                                       deterministic=deterministic)

        # supervised learning
        sl_lower_bound, sl_loss, sl_params =\
            self._supervised_learning(x_l, y, l, deterministic=deterministic)
        lower_bound = T.concatenate([lower_bound, sl_lower_bound], axis=-1)
        loss = loss + self.sl_rate * sl_loss

        params += sl_params

        return lower_bound, loss, params
