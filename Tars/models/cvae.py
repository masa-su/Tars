import theano
import theano.tensor as T

from ..utils import log_mean_exp
from ..models.vae import VAE


class CVAE(VAE):

    def _set_test(self, type_p="normal", missing=False):
        # set inputs
        x = self.q.inputs
        l = T.iscalar("l")
        k = T.iscalar("k")

        inputs = x + [l, k]
        lower_bound = self._vr_bound_test(x, l, k, missing, True)
        self.lower_bound_test = theano.function(inputs=inputs,
                                                outputs=lower_bound,
                                                on_unused_input='ignore')

    def _set_test(self, type_p="marginal", missing=False):
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
                x, l, k, missing)

        self.lower_bound_test = theano.function(inputs=inputs,
                                                outputs=lower_bound,
                                                on_unused_input='ignore')

    def test(self, test_set, l=1, k=1, n_batch=None, verbose=True,
             type_p="normal", missing=False):
        if len(test_set) > 1 and type_p == "conditional":
            self._set_test(type_p, missing)

        return super(CVAE, self).test(test_set, l=l, k=k,
                                      n_batch=n_batch, verbose=verbose)

    def _vr_bound_test(self, x, l, k, missing=False, deterministic=False):
        n_x = x[0].shape[0]
        rep_x = [T.extra_ops.repeat(_x, l * k, axis=0) for _x in x]

        if missing:
            (n_z, z_dim) = self.p.inputs[0].shape
            rep_z = self.q.sample_given_x(rep_x, deterministic=True)[-1]
            prior_rep_z = self.prior.sample(rep_z.shape)
            log_iw = self._log_cd_importance_weight([rep_x, prior_rep_z])

        else:
            samples = self.q.sample_given_x(rep_x, deterministic=True)
            log_iw = self._log_reconstruct_weight(samples, deterministic=True)

        log_iw_matrix = T.reshape(log_iw, (n_x * l, k))
        log_likelihood = log_mean_exp(
            log_iw_matrix, axis=1, keepdims=True)
        log_likelihood = log_likelihood.reshape((x[0].shape[0], l))
        log_likelihood = T.mean(log_likelihood, axis=1)

        return log_likelihood

    def _log_reconstruct_weight(self, samples, deterministic=False):
        """
        Paramaters
        ----------
        samples : list
           [[x0,x1,...],z1,z2,...,zn]

        Returns
        -------
        log_iw : array, shape (n_samples*k)
           Estimated log likelihood.
           log p(x1|z1,z2,...,zn)
        """

        log_iw = 0
        # log p(x1|z1,...)
        p_samples, prior_samples = self._inverse_samples(
            samples, return_prior=True)
        p_log_likelihood = self.p.log_likelihood_given_x(
            p_samples, deterministic=deterministic)

        log_iw += p_log_likelihood
        # log p(z1,,z2,...|zn)
        if self.prior_mode == "MultiPrior":
            log_iw += self.prior.log_likelihood_given_x(prior_samples,
                                                        add_prior=False)

        return log_iw

    def _log_cd_importance_weight(self, samples, deterministic=True):
        """
        Paramaters
        ----------
        samples : list
           [[x0,x1,...],z1,z2,...,zn]

        Returns
        -------
        log_iw : array, shape (n_samples*k)
           Estimated log likelihood.
           log p(x0|z,x1)
        """

        log_iw = 0
        # log p(x0|z1,...,z1)
        p_samples, prior_samples = self._inverse_samples(
            samples, return_prior=True)
        p_log_likelihood = self.p.log_likelihood_given_x(
            p_samples, deterministic=deterministic)

        log_iw += p_log_likelihood
        # log p(z1,,z2,...|zn)
        if self.prior_mode == "MultiPrior":
            log_iw += self.prior.log_likelihood_given_x(prior_samples,
                                                        add_prior=False)

        return log_iw
