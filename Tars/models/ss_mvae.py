import theano.tensor as T
import lasagne
import numpy as np

from ..utils import tolist, log_mean_exp
from .ss_vae import SS_VAE


class SS_MVAE(SS_VAE):
    def __init__(self, q, p, f, prior=None,
                 n_batch=100, n_batch_u=100,
                 optimizer=lasagne.updates.adam,
                 optimizer_params={},
                 sum_classes=False,
                 clip_grad=None, max_norm_constraint=None,
                 regularization_penalty=None,
                 seed=1234):

        super(SS_MVAE,
              self).__init__(q, p, f, prior=prior,
                             n_batch=n_batch, n_batch_u=n_batch_u,
                             optimizer=optimizer, optimizer_params=optimizer_params,
                             sum_classes=sum_classes,
                             clip_grad=clip_grad, max_norm_constraint=max_norm_constraint,
                             regularization_penalty=regularization_penalty,
                             seed=seed)

    def _vr_bound(self, x, l=1, k=1,
                  iw_alpha=0, deterministic=False, y=None):

        if y is None:
            supervised = False
        else:
            supervised = True
        rep_x = [T.extra_ops.repeat(_x, l * k, axis=0) for _x in x]

        if supervised is False:
            # y ~ q(y|x,w)
            rep_y = self.f.sample_given_x(rep_x,
                                          deterministic=deterministic)[-1:]
        else:
            rep_y = [T.extra_ops.repeat(_y, l * k, axis=0) for _y in y]

        # z ~ q(z|x,w,y)

        z = self.q.sample_given_x(rep_x+rep_y,
                                  deterministic=deterministic)[-1:]
        # q_samples = [[x,w],z]
        q_samples = [rep_x]+z

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
            p_params += self.p[i].get_params()
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
           [[x,w],z]

        samples : y
           [y]

        Returns
        -------
        log_iw : array, shape (n_samples)
           Estimated log likelihood.
           supervised=True:
             log p(x,w,y,z)/q(y,z|x,w)
           supervised=False:
             log p(x,w,y,z)/q(z|x,w,y)
        """

        # samples : [[x,w],z]
        log_iw = 0

        # log q(z|x,w,y)
        # samples_y : [[x,w,y],z]
        samples_y = [tolist(samples[0])+y, samples[-1]]
        q_log_likelihood = self.q.log_likelihood_given_x(
            samples_y, deterministic=deterministic)

        if supervised is False:
            # log q(y|x,w)
            # _samples : [[x,w],y]
            _samples = [tolist(samples[0])]+y
            q_log_likelihood += self.f.log_likelihood_given_x(
                _samples, deterministic=deterministic)

        # log p(x|z,y), log p(x|z,y)
        # samples : [[z,y],x], [[z,y],w]
        p_log_likelihood_all = []
        for i, p in enumerate(self.p):
            _samples = self._select_input(samples, [i])
            _samples_y = [tolist(_samples[0])+y, _samples[-1]]
            p_samples, prior_samples = self._inverse_samples(
                _samples_y, return_prior=True)
            p_log_likelihood = self.p[i].log_likelihood_given_x(
                p_samples, deterministic=deterministic)
            p_log_likelihood_all.append(p_log_likelihood)
        log_iw += sum(p_log_likelihood_all) - q_log_likelihood

        if self.prior_mode == "MultiPrior":
            log_iw += self.prior.log_likelihood_given_x(prior_samples)
        else:
            log_iw += self.prior.log_likelihood(prior_samples)
        #log_iw += 1. / np.float32(10)  # Categorical prior distribution

        return log_iw
