import numpy as np
import theano
import theano.tensor as T

from . import VAE
from ..utils import t_repeat, beta


# https://github.com/enalisnick/stick-breaking_dgms
class SB_VAE(VAE):
    def __init__(self, q, p, n_batch, optimizer,
                 l=1, k=1, random=1234,
                 prior_alpha=1., prior_beta=5.,
                 m=10, gamma=0.57721):
        self.gamma = gamma
        self.m = m
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta

        super(SB_VAE, self).__init__(q, p, n_batch, optimizer,
                                     l, k, None, random)

    def lowerbound(self):
        x = self.q.inputs
        annealing_beta = T.fscalar("beta")

        a, b = self.q.fprop(x, self.srng, deterministic=False)
        kl = self._kumaraswamy_beta_kl(a, b).mean()
        rep_x = [t_repeat(_x, self.l, axis=0) for _x in x]
        v = self.q.sample_given_x(rep_x, self.srng, deterministic=False)
        v[-1] = self._stick_breaking_process(v[-1])

        inverse_v = self.inverse_samples(v)
        loglike = self.p.log_likelihood_given_x(inverse_v).mean()

        lowerbound = [-kl, loglike]
        loss = -(loglike-annealing_beta*kl)

        q_params = self.q.get_params()
        p_params = self.p.get_params()
        params = q_params + p_params

        updates = self.optimizer(loss, params)
        self.lowerbound_train = theano.function(
            inputs=x+[annealing_beta],
            outputs=lowerbound,
            updates=updates,
            on_unused_input='ignore')

        self.hidden = theano.function(
            inputs=x,
            outputs=v[-1],
            on_unused_input='ignore')

    def p_sample_mean_given_x(self):
        x = self.p.inputs
        x[0] = self._stick_breaking_process(x[0])
        samples = self.p.sample_mean_given_x(x, self.srng, deterministic=True)
        self.p_sample_mean_x = theano.function(
            inputs=x, outputs=samples[-1], on_unused_input='ignore')

        samples = self.p.sample_given_x(x, self.srng, deterministic=True)
        self.p_sample_x = theano.function(
            inputs=x, outputs=samples[-1], on_unused_input='ignore')

        samples = self.p.fprop(x, self.srng, deterministic=True)
        self.p_sample_meanvar_x = theano.function(
            inputs=x, outputs=samples, on_unused_input='ignore')

    def q_sample_mean_given_x(self):
        x = self.q.inputs
        samples = self.q.sample_given_x(x, self.srng, deterministic=True)
        samples[-1] = self._stick_breaking_process(samples[-1])

        self.q_sample_x = theano.function(
            inputs=x, outputs=samples[-1], on_unused_input='ignore')

    def log_marginal_likelihood(self, x, l):
        n_x = x[0].shape[0]
        rep_x = [t_repeat(_x, l, axis=0) for _x in x]

        a, b = self.q.fprop(x, self.srng, deterministic=True)
        kl = self._kumaraswamy_beta_kl(a, b).mean()

        samples = self.q.sample_given_x(rep_x, self.srng)
        samples[-1] = self._stick_breaking_process(samples[-1])

        inverse_samples = self.inverse_samples(samples)
        log_iw = self.p.log_likelihood_given_x(inverse_samples)
        log_iw_matrix = T.reshape(log_iw, (n_x, l))
        log_marginal_estimate = kl + T.mean(log_iw_matrix, axis=1)

        return log_marginal_estimate

    def log_importance_weight(self, samples):
        """
        Paramaters
        ----------
        samples : list
           [[x,y,...],z1,z2,...,zn]

        Returns
        -------
        log_iw : array, shape (n_samples*k)
           Estimated log likelihood.
           log p(x,z1,z2,...,zn|y,...)/q(z1,z2,...,zn|x,y,...)
        """

        log_iw = 0

        # log q(z1,z2,...,zn|x,y,...)
        # samples : [[x,y,...],z1,z2,...,zn]
        q_log_likelihood = self.q.log_likelihood_given_x(samples)

        # Stick breaking process (SBP)
        samples[-1] = self._stick_breaking_process(samples[-1])

        # log p(x|z1,z2,...,zn,y,...)
        # inverse_samples : [[zn,y,...],zn-1,...,x]
        inverse_samples = self.inverse_samples(samples)
        p_log_likelihood = self.p.log_likelihood_given_x(inverse_samples)

        log_iw += p_log_likelihood - q_log_likelihood
        log_iw += self.prior.log_likelihood(samples[-1])

        return log_iw

    def _stick_breaking_process(self, v):
        (n_batch, n_dim) = v.shape

        def segment(v, stick_segment, remaining_stick):
            stick_segment = v * remaining_stick
            remaining_stick *= (1 - v)
            return stick_segment, remaining_stick

        stick_segment = T.zeros((n_batch,))
        remaining_stick = T.ones((n_batch,))

        (stick_segments, remaining_sticks), updates\
            = theano.scan(fn=segment,
                          sequences=v.T[:-1],
                          outputs_info=[stick_segment, remaining_stick])

        return T.concatenate([stick_segments.T,
                              remaining_sticks[-1, :][:, np.newaxis]],
                             axis=1)

    def _kumaraswamy_beta_kl(self, a, b):
        def taylor(i, a, b):
            return 1./(i + a*b) * beta(1/a, b)

        kl, updates = theano.scan(fn=taylor,
                                  sequences=T.arange(self.m),
                                  non_sequences=[a, b])
        kl *= (self.prior_beta - 1)*b

        # Because T.psi haven't implemented yet.
        psi = T.log(b) - 1./(2 * b) - 1./(12 * b**2)
        kl += (a-self.prior_alpha)/a * (-self.gamma - psi - 1/b)
        kl += T.log(a*b) + T.log(beta(self.prior_alpha, self.prior_beta))
        kl += -(b - 1)/b

        return kl
