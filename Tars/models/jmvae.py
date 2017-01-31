from copy import copy

import theano
import theano.tensor as T
import lasagne

from ..utils import tolist, log_mean_exp
from ..distributions.estimate_kl import analytical_kl
from . import VAE


class JMVAE(VAE):

    def __init__(self, q, p,
                 n_batch=100, optimizer=lasagne.updates.adam,
                 optimizer_params={},
                 clip_grad=None, max_norm_constraint=None,
                 train_iw=False, test_iw=True, seed=1234):
        super(JMVAE,
              self).__init__(q, p,
                             n_batch=n_batch,
                             optimizer=optimizer,
                             optimizer_params=optimizer_params,
                             clip_grad=clip_grad,
                             max_norm_constraint=max_norm_constraint,
                             train_iw=train_iw, test_iw=test_iw,
                             iw_alpha=0, seed=seed)

    def _set_test(self, type_p="marginal", missing=False,
                  index=[0], sampling_n=1, missing_resample=False):
        # set inputs
        x = self.q.inputs
        l = T.iscalar("l")
        k = T.iscalar("k")

        if type_p == "joint":
            if self.test_iw:
                inputs = x + [l, k]
                lower_bound, _, _ = self._vr_bound(x, l, k, 0, True)
            else:
                inputs = x + [l]
                lower_bound, _, _ = self._elbo(x, l, 1, True)
                lower_bound = T.sum(lower_bound, axis=1)
        else:
            inputs = x + [l, k]
            lower_bound = self._vr_bound_test(
                x, l, k, index, type_p, missing, sampling_n, missing_resample)

        self.lower_bound_test = theano.function(inputs=inputs,
                                                outputs=lower_bound,
                                                on_unused_input='ignore')

    def test(self, test_set, l=1, k=1, index=[0], sampling_n=1,
             missing_resample=False, type_p="joint",
             missing=False, n_batch=None, verbose=True):

        self._set_test(type_p, missing, index, sampling_n, missing_resample)
        return super(JMVAE, self).test(test_set,
                                       l, k, n_batch, verbose)

    def _elbo(self, x, l, annealing_beta, deterministic=False):

        kl_divergence = analytical_kl(self.q, self.prior,
                                      given=[x, None],
                                      deterministic=deterministic)
        z = self.q.sample_given_x(x, repeat=l,
                                  deterministic=deterministic)

        log_likelihood_all = []
        p_params = []
        for i, p in enumerate(self.p):
            inverse_z = self._inverse_samples(self._select_input(z, [i]))
            log_likelihood = self.p[i].log_likelihood_given_x(
                inverse_z, deterministic=deterministic)
            log_likelihood_all.append(log_likelihood)
            p_params += self.p[i].get_params()

        lower_bound = T.stack([-kl_divergence] + log_likelihood_all, axis=-1)
        loss = -T.mean(sum(log_likelihood_all)
                       - annealing_beta * kl_divergence)

        q_params = self.q.get_params()
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
                _rep_x = self._select_input([rep_x], index, set_zeros=True)[0]
                samples = self.q.sample_given_x(_rep_x, deterministic=True)
                samples = self._select_input(samples, inputs=rep_x)
                log_iw = self._log_mg_missing_importance_weight(
                    samples, index, deterministic=True)

            elif type_p == "conditional":
                # rep_x:[x0,x1] -> _rep_x:[0,x1]
                rv_index = self._reverse_index(index)
                _rep_x = self._select_input(
                    [rep_x], rv_index, set_zeros=True)[0]

                if missing_resample:
                    for _ in range(sampling_n-1):
                        samples = self.q.sample_given_x(_rep_x, deterministic=True)[-1]
                        for j in index:
                            _rep_x[j] = self.p[j].sample_given_x([samples])[-1]

                samples = self.q.sample_given_x(_rep_x, deterministic=True)
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

    def _log_importance_weight(self, samples, deterministic=True):
        """
        Paramaters
        ----------
        samples : list
           [[x0,x1,...],z1,z2,...,zn]

        Returns
        -------
        log_iw : array, shape (n_samples)
           Estimated log likelihood.
           log p(x0,x1,...,z1,z2,...,zn)/q(z1,z2,...,zn|x0,x1,...)
        """

        log_iw = 0

        # log q(z1,z2,...,zn|x0,x1,...)
        # samples : [[x0,x1,...],z1,z2,...,zn]
        q_log_likelihood = self.q.log_likelihood_given_x(
            samples, deterministic=deterministic)

        # log p(x|z1,z2,...,zn)
        # inverse_samples0 : [zn,zn-1,...,x]
        p_log_likelihood_all = []
        for i, p in enumerate(self.p):
            inverse_samples = self._inverse_samples(
                self._select_input(samples, [i]))
            p_log_likelihood = self.p[i].log_likelihood_given_x(
                inverse_samples, deterministic=deterministic)
            p_log_likelihood_all.append(p_log_likelihood)

        log_iw += sum(p_log_likelihood_all) - q_log_likelihood
        log_iw += self.prior.log_likelihood(samples[-1])

        return log_iw

    def _log_selected_importance_weight(self, samples, index=[0],
                                        deterministic=True):
        """
        Paramaters
        ----------
        samples : list
           [[x0,x1,...],z1,z2,...,zn]

        index : list

        Returns
        -------
        log_iw : array, shape (n_samples*k)
           Estimated log likelihood.
           log p(x[index],z1,z2,...,zn)/q(z1,z2,...,zn|x0,x1,...)
        """

        log_iw = 0

        # log q(z1,z2,...,zn|x0,x1,...)
        # samples : [[x0,x1,...],z1,z2,...,zn]
        q_log_likelihood = self.q.log_likelihood_given_x(
            samples, deterministic=deterministic)

        # log p(x[index]|z1,z2,...,zn)
        # inverse_samples0 : [zn,zn-1,...,x[index]]
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

        # samples : [[x0,0,...],z1,z2,...,zn]
        samples = self._select_input(samples, index, set_zeros=True)

        # log q(z1,z2,...,zn|x0,0,...)
        # samples : [[x0,0,...],z1,z2,...,zn]
        q_log_likelihood = self.q.log_likelihood_given_x(
            samples, deterministic=deterministic)

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

    def _log_cd_importance_weight(self, samples, index=[0],
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
           log p(x[index]|z1,z2,...,zn)
        """

        log_iw = 0
        # log p(x[index]|z1,z2,...,zn)
        p_log_likelihood_all = []
        for i in index:
            inverse_samples = self._inverse_samples(
                self._select_input(samples, [i]))
            p_log_likelihood = self.p[i].log_likelihood_given_x(
                inverse_samples, deterministic=deterministic)
            p_log_likelihood_all.append(p_log_likelihood)

        log_iw += sum(p_log_likelihood_all)

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
