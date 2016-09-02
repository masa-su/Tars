import theano

from . import AE


class MAE(AE):

    def __init__(self, q, p, n_batch, optimizer, random=1234):
        super(MAE, self).__init__(q, p, n_batch, optimizer, random)

    def lowerbound(self, random):
        x = self.q.inputs
        z = self.q.fprop(x, deterministic=False)

        inverse_z = self.inverse_samples([[x[0]], z])
        loglike0 = self.p[0].log_likelihood_given_x(inverse_z).mean()

        inverse_z = self.inverse_samples([[x[1]], z])
        loglike1 = self.p[1].log_likelihood_given_x(inverse_z).mean()
        loglike = loglike0 + loglike1

        q_params = self.q.get_params()
        p0_params = self.p[0].get_params()
        p1_params = self.p[1].get_params()
        params = q_params + p0_params + p1_params

        updates = self.optimizer(-loglike, params)
        self.lowerbound_train = theano.function(
            inputs=x,
            outputs=loglike,
            updates=updates,
            on_unused_input='ignore')

    def p_sample_mean_given_x(self):
        x = self.p[0].inputs
        samples = self.p[0].sample_mean_given_x(x, False)
        self.p0_sample_mean_x = theano.function(
            inputs=x, outputs=samples[-1], on_unused_input='ignore')

        x = self.p[1].inputs
        samples = self.p[1].sample_mean_given_x(x, False)
        self.p1_sample_mean_x = theano.function(
            inputs=x, outputs=samples[-1], on_unused_input='ignore')

    def log_marginal_likelihood(self, x):
        z = self.q.fprop(x, deterministic=True)
        inverse_z = self.inverse_samples([[x[0]], z])
        loglike0 = self.p[0].log_likelihood_given_x(inverse_z)

        inverse_z = self.inverse_samples([[x[1]], z])
        loglike1 = self.p[1].log_likelihood_given_x(inverse_z)

        log_marginal_estimate = loglike0 + loglike1

        return log_marginal_estimate
