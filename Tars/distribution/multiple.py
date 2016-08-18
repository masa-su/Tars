import theano.tensor as T

from ..utils import tolist


class Concatenate(object):
    """
    This distribution is used to concatenate different distributions in
    their feature axis. Therefore, we can handle multiple distributions
    as one distribution when sampling from them or estimating their
    log-likelihood. However, every distribution must have same given
    variables.

    Samples
    -------
    distributions : list
       Contain multiple distributions.

    Examples
    --------
    >>> from Tars.distribution import Concatenate, Gaussian, Bernoulli
    >>> gauss = Gaussian(mean, var, given=[x])
    >>> bernoulli = Bernoulli(mean, given=[x])
    >>> concat = Concatenate([gauss, bernoulli])
    """

    def __init__(self, distributions):
        self.distributions = distributions
        self.outputs_dim = 0
        for d in self.distributions:
            if self.distributions[0].given != d.given:
                raise ValueError("Every distribution must have same "
                                 "given variables")
            self.outputs_dim += d.get_output_shape()[-1]

        self.inputs = self.distributions[0].inputs

    def get_params(self):
        params = []
        for d in self.distributions:
            params += d.get_params()
        return params

    def fprop(self, x, *args, **kwargs):
        """
        Samples
        -------
        x : list
           This contains Theano variables. The number of them must be
           same as 'distributions'.
        """

        samples = []
        for d in self.distributions:
            samples.append(d.fprop(x, *args, **kwargs))
        return T.concatenate(samples, axis=-1)

    def sample_given_x(self, x, srng, **kwargs):
        samples = []
        for d in self.distributions:
            samples.append(
                d.sample_given_x(x, srng, **kwargs)[-1])
        return [x, T.concatenate(samples, axis=-1)]

    def sample_mean_given_x(self, x, *args, **kwargs):
        samples = []
        for d in self.distributions:
            samples.append(d.sample_mean_given_x(
                x, *args, **kwargs)[-1])
        return [x, T.concatenate(samples, axis=-1)]

    def log_likelihood_given_x(self, samples, **kwargs):
        """
        Samples
        -------
        samples : list
           This contains 'x', which has Theano variables, and test sample.
           The dimension of test sample must be same as output_dim.

        Returns
        -------
        Theano variable, shape (n_samples,)
           A log-likelihood, p(sample|x)
        """

        x, sample = samples

        if sample.shape[-1] != self.outputs_dim:
            raise ValueError("The dimension of test sample must be same as "
                             "output_dim.")

        loglikes = 0
        start = 0
        for d in enumerate(self.distributions):
            output_dim = d.get_output_shape()[-1]
            loglikes += d.log_likelihood_given_x(
                [x, sample[:, start:start+output_dim]],
                **kwargs)
            start += output_dim
        return loglikes


class Multilayer(object):
    """
    This distribution is used to stack multiple distiributions, that is
    p(x|z) = p(x|z1)p(z1|z2)...p(zn-1|zn). If the distributions are
    approximate distributions, then a corresponding stacked distribution
    becomes like q(z|x) = q(zn|zn-1)...q(z2|z1)q(z1|x). If the stacked
    distribution is conditioned on y, then the corresponding mean field
    approximation becomes like p(x|z,y) = p(x|z1)p(z1|z2)...p(zn-1|zn,y),
    or q(z|x,y) = q(zn|zn-1)...q(z2|z1)q(z1|x,y).
    So far, each distribution except first layer cannot have conditioned
    variables more than two.

    Samples
    -------
    distributions : list
       Contain multiple distributions.

    Examples
    --------
    >>> from Tars.distribution import Multilayer, Gaussian, Bernoulli
    >>> gauss = Gaussian(mean, var, given=[x])
    >>> bernoulli = Bernoulli(mean, given=[z])
    >>> multi = Multilayer([gauss, bernoulli])
    """

    def __init__(self, distributions):
        self.distributions = distributions
        self.inputs = self.distributions[0].inputs
        for i, d in enumerate(distributions[1:]):
            if len(d.given) != 1:
                raise ValueError("So far, each distribution except first "
                                 "layer cannot have conditioned variables "
                                 "more than two.")

            if distributions[i].get_output_shape() != d.given[0].shape:
                raise ValueError("An output's shape of a distribution must be "
                                 "same as an input's shape of the next layer "
                                 "distribution.")

    def get_params(self):
        params = []
        for d in self.distributions:
            params += d.get_params()
        return params

    def _sampling(self, x, srng, **kwargs):
        """
        Paramaters
        ----------
        x : list
           This contains Theano variables.

        srng : theano.sandbox.MRG_RandomStreams

        Returns
        -------
        list
           This contains 'x' and samples, such as [x,z1,...,zn-1].
        """

        samples = [x]
        for i, d in enumerate(self.distributions[:-1]):
            sample = d.sample_given_x(
                samples[i], srng, **kwargs)
            samples.append(sample[-1])
        return samples

    def fprop(self, x, srng, **kwargs):
        """
        Paramaters
        ----------
        x : list
           This contains Theano variables.

        srng : theano.sandbox.MRG_RandomStreams

        Returns
        -------
        mean : Theano variable
            The output of this distribution.
        """

        samples = self._sampling(x, srng, **kwargs)
        mean = self.distributions[-1].fprop(
            tolist(samples[-1]), **kwargs)
        return mean

    def sample_given_x(self, x, srng, **kwargs):
        """
        Paramaters
        --------
        x : list
           This contains Theano variables, which must to correspond
           to 'given' of first layer distibution.

        srng : theano.sandbox.MRG_RandomStreams

        Returns
        --------
        list
           This contains 'x' and samples, such as [x,z1,...,zn].
        """

        samples = self._sampling(x, srng, **kwargs)
        samples += self.distributions[-1].sample_given_x(
            tolist(samples[-1]), srng, **kwargs)[-1:]
        return samples

    def sample_mean_given_x(self, x, srng, **kwargs):
        """
        Paramaters
        --------
        x : list
           This contains Theano variables, which must to correspond
           to 'given'.

        Returns
        --------
        list
           This contains 'x', samples, and a mean value of sample,
           such as [x,z1,...,zn_mean]
        """

        mean = self._sampling(x, srng, **kwargs)
        mean += self.distributions[-1].sample_mean_given_x(
            tolist(mean[-1]), **kwargs)[-1:]
        return mean

    def log_likelihood_given_x(self, samples, deterministic=False):
        """
        Paramaters
        --------
        samples : list
           This contains 'x', which has Theano variables, and test samples,
           such as z1, z2,...,zn.

        deterministic : bool

        Returns
        --------
        Theano variable, shape (n_samples,)
           log_likelihood (q) : log_q(z1|[x,y,...])+...+log_q(zn|zn-1)
           log_likelihood (p) : log_p(zn-1|[zn,y,...])+...+log_p(x|z1)
        """

        all_log_likelihood = 0
        for x, sample, d in zip(samples, samples[1:], self.distributions):
            log_likelihood = d.log_likelihood_given_x([tolist(x), sample])
            all_log_likelihood += log_likelihood
        return all_log_likelihood
