from copy import copy

import theano
import theano.tensor as T
import lasagne

from ..utils import tolist, log_mean_exp
from ..distributions.estimate_kl import analytical_kl
from . import JMVAE


class JMVAE_KL(JMVAE):

    def __init__(self, q, p, pseudo_q, gamma=1,
                 n_batch=100, optimizer=lasagne.updates.adam,
                 optimizer_params={},
                 clip_grad=None, max_norm_constraint=None,
                 test_iw=True, seed=1234):
        self.pseudo_q = pseudo_q
        self.gamma = gamma

        super(JMVAE_KL,
              self).__init__(q, p,
                             n_batch=n_batch,
                             optimizer=optimizer,
                             optimizer_params=optimizer_params,
                             clip_grad=clip_grad,
                             max_norm_constraint=max_norm_constraint,
                             train_iw=False, test_iw=test_iw,
                             seed=seed)

    def _elbo(self, x, l, annealing_beta, deterministic=False):
        lower_bound, loss, params = \
            super(JMVAE_KL, self)._elbo(x, l, annealing_beta, deterministic=False)

        # ---penalty
        kl_all = []
        pseudo_q_params = []
        for i, _pseudo_q in enumerate(self.pseudo_q):
            kl_all.append(analytical_kl(self.q, _pseudo_q, given=[x, [x[i]]],
                                        deterministic=False))
            pseudo_q_params += _pseudo_q.get_params()

        kl_all = T.stack(kl_all, axis=-1)
        lower_bound = T.concatenate([lower_bound, kl_all], axis=-1)
        loss += self.gamma * T.mean(kl_all)

        params += pseudo_q_params

        return lower_bound, loss, params

    def _vr_bound_test(self, x, l, k, index=[0], type_p="marginal",
                       missing=False, sampling_n=1, missing_resample=False):
        """
        Paramaters
        ----------
        x : TODO

        l : TODO

        k : TODO

        type_p : {'conditional', 'marginal'}
           Specifies the type of the log likelihood.

        Returns
        --------
        log_marginal_estimate : array, shape (n_samples)
           Estimated log likelihood.
        """

        n_x = x[0].shape[0]
        rep_x = [T.extra_ops.repeat(_x, l * k, axis=0) for _x in x]

        if type_p not in ['marginal', 'conditional']:
            raise ValueError("type_p must be one of {"
                             "'marginal', 'conditional'}, got %s." % type_p)

        if missing:
            if type_p == "marginal":
                _rep_x = self._select_input([rep_x], index)[0]
                samples = self.pseudo_q[index[0]].sample_given_x(_rep_x, deterministic=True)
                samples = self._select_input(samples, inputs=rep_x)
                log_iw = self._log_mg_missing_importance_weight(
                    samples, index, deterministic=True)

            elif type_p == "conditional":
                # rep_x:[x0,x1] -> _rep_x:[0,x1]
                rv_index = self._reverse_index(index)
                _rep_x = self._select_input(
                    [rep_x], rv_index)[0]
                samples = self.pseudo_q[rv_index[0]].sample_given_x(_rep_x, deterministic=True)
                samples = self._select_input(samples, inputs=rep_x)
                log_iw = self._log_cd_importance_weight(
                    samples, index, deterministic=True)

        else:
            samples = self.q.sample_given_x(rep_x, deterministic=True)
            if type_p == "marginal":
                log_iw = self._log_selected_importance_weight(
                    samples, index, deterministic=True)
            else:
                log_iw = self._log_cd_importance_weight(
                    samples, deterministic=True)

        log_iw_matrix = T.reshape(log_iw, (n_x * l, k))
        log_likelihood = log_mean_exp(
            log_iw_matrix, axis=1, keepdims=True)
        log_likelihood = log_likelihood.reshape((x[0].shape[0], l))
        log_likelihood = T.mean(log_likelihood, axis=1)

        return log_likelihood

    def _log_mg_missing_importance_weight(self, samples, index=[0],
                                          deterministic=True):
        """
        Paramaters
        ----------
        samples : list
           [[x0,x1,...],z1,z2,...,zn]

        Returns
        -------
        log_iw : array, shape (n_samples*k)
           Estimated log likelihood.
           log p(x[index],z1,z2,...,zn)/q(z1,z2,...,zn|x[index])
        """

        log_iw = 0

        # log q(z1,z2,...,zn|x0,0,...)
        # _samples : [[x0,0,...],z1,z2,...,zn]
        _samples = self._select_input(samples, [index[0]])
        q_log_likelihood = self.pseudo_q[0].log_likelihood_given_x(
            _samples, deterministic=deterministic)

        # log p(x0|z1,z2,...,zn)
        # inverse_samples0 : [zn,zn-1,...,x0]
        p_log_likelihood_all = []
        for i in index:
            inverse_samples = self._inverse_samples(
                self._select_input(samples, [i]))
            p_log_likelihood = self.p[i].log_likelihood_given_x(
                inverse_samples, deterministic=deterministic)
            p_log_likelihood_all.append(p_log_likelihood)

        log_iw += sum(p_log_likelihood_all) - q_log_likelihood
        log_iw += self.prior.log_likelihood(samples[-1])

        return log_iw
