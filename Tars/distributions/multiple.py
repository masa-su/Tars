import theano
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
                [x, sample[:, start:start + output_dim]],
                **kwargs)
            start += output_dim
        return loglikes


class MultiDistributions(object):
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
    >>> from Tars.distribution import MultiDistributions
    >>> from Tars.distribution import Gaussian, Bernoulli
    >>> gauss = Gaussian(mean, var, given=[x])
    >>> bernoulli = Bernoulli(mean, given=[z])
    >>> multi = MultiDistributions([gauss, bernoulli])
    """

    def __init__(self, distributions, approximate=True):
        self.distributions = distributions
        self.given = self.distributions[0].given
        self.inputs = self.distributions[0].inputs
        self.output = self.distributions[-1].output
        self.get_input_shape = self.distributions[0].get_input_shape
        self.get_output_shape = self.distributions[-1].get_output_shape
        self.approximate = approximate
        self._set_theano_func()

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

    def _sample(self, x, layer_id, repeat=1, **kwargs):
        """
        Paramaters
        ----------
        x : list
           This contains Theano variables.

        Returns
        -------
        list
           This contains 'x' and samples, such as [x,z1,...,zn-1].
        """

        samples = [[T.extra_ops.repeat(_x, repeat, axis=0) for _x in x]]
        for i, d in enumerate(self.distributions[:layer_id]):
            sample = d.sample_given_x(
                tolist(samples[i]), **kwargs)
            samples.append(sample[-1])
        return samples

    def _sample_mean(self, x, layer_id, **kwargs):
        """
        Paramaters
        ----------
        x : list
           This contains Theano variables.

        Returns
        -------
        list
           This contains 'x' and samples, such as [x,z1,...,zn-1].
        """
        samples = [x]
        for i, d in enumerate(self.distributions[:layer_id]):
            sample = d.sample_mean_given_x(
                tolist(samples[i]), **kwargs)
            samples.append(sample[-1])
        return samples

    def _approx_sample(self, x, layer_id, repeat=1, **kwargs):
        """
        Paramaters
        ----------
        x : list
           This contains Theano variables.

        Returns
        -------
        list
           This contains 'x' and samples, such as [x,z1,...,zn-1].
        """

        mean = x
        samples = [[T.extra_ops.repeat(_x, repeat, axis=0) for _x in x]]
        for d in self.distributions[:layer_id]:
            sample = d.sample_given_x(
                tolist(mean), repeat=repeat, **kwargs)
            samples.append(sample[-1])
            mean = d.sample_mean_given_x(
                tolist(mean), **kwargs)[-1]

        return samples, mean

    def fprop(self, x, layer_id=-1, *args, **kwargs):
        """
        Paramaters
        ----------
        x : list
           This contains Theano variables.

        Returns
        -------
        mean : Theano variable
            The output of this distribution.
        """
        if self.approximate:
            output = self._sample_mean(x, layer_id, **kwargs)[-1]
        else:
            output = self._sample(x, layer_id, **kwargs)[-1]
        mean = self.distributions[layer_id].fprop(
            tolist(output), *args, **kwargs)
        return mean

    def sample_given_x(self, x, layer_id=-1, repeat=1, **kwargs):
        """
        Paramaters
        --------
        x : list
           This contains Theano variables, which must to correspond
           to 'given' of first layer distibution.

        repeat : int or thenao variable

        Returns
        --------
        list
           This contains 'x' and samples, such as [x,z1,...,zn].
        """

        if self.approximate:
            samples, mean = self._approx_sample(x, layer_id,
                                                repeat=repeat, **kwargs)
            samples += self.distributions[layer_id].sample_given_x(
                tolist(mean), repeat=repeat, **kwargs)[-1:]
        else:
            samples = self._sample(x, layer_id, repeat=repeat, **kwargs)
            samples += self.distributions[layer_id].sample_given_x(
                tolist(samples[-1]), repeat=repeat, **kwargs)[-1:]
        return samples

    def sample_mean_given_x(self, x, layer_id=-1, *args, **kwargs):
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

        if self.approximate:
            mean = self._sample_mean(x, layer_id, **kwargs)
        else:
            mean = self._sample(x, layer_id, **kwargs)
        mean += self.distributions[layer_id].sample_mean_given_x(
            tolist(mean[-1]), *args, **kwargs)[-1:]
        return mean

    def log_likelihood_given_x(self, samples, last_layer=False, **kwargs):
        """
        Paramaters
        --------
        samples : list
           This contains 'x', which has Theano variables, and test samples,
           such as z1, z2,...,zn.

        Returns
        --------
        Theano variable, shape (n_samples,)
           log_likelihood (q) : log_q(z1|[x,y,...])+...+log_q(zn|zn-1)
           log_likelihood (p) : log_p(zn-1|[zn,y,...])+...+log_p(x|z1)
        """
        if last_layer:
            d = self.distributions[-1]
            all_log_likelihood = d.log_likelihood_given_x([tolist(samples[0]),
                                                           samples[1]],
                                                          **kwargs)
        else:
            all_log_likelihood = 0
            for x, sample, d in zip(samples, samples[1:], self.distributions):
                log_likelihood = d.log_likelihood_given_x([tolist(x), sample],
                                                          **kwargs)
                all_log_likelihood += log_likelihood

        return all_log_likelihood

    def _set_theano_func(self):
        x = self.inputs
        samples = self.fprop(x, layer_id=-1, deterministic=True)
        self.np_fprop = theano.function(inputs=x,
                                        outputs=samples,
                                        on_unused_input='ignore')

        samples = self.sample_mean_given_x(x, layer_id=-1, deterministic=True)
        self.np_sample_mean_given_x = theano.function(
            inputs=x, outputs=samples[-1], on_unused_input='ignore')

        samples = self.sample_given_x(x, layer_id=-1, deterministic=True)
        self.np_sample_given_x = theano.function(
            inputs=x, outputs=samples[-1], on_unused_input='ignore')


class MultiPriorDistributions(MultiDistributions):
    """
    p(z) = p(zn,z'n)p(zn-1|zn,z'n)...p(z1|z2).

    Samples
    -------
    distributions : list
       Contain multiple distributions.

    Examples
    --------
    >>> from Tars.distribution import MultiPriorDistributions
    >>> from Tars.distribution import Gaussian, Bernoulli
    >>> gauss = Gaussian(mean, var, given=[z2])
    >>> bernoulli = Bernoulli(mean, given=[z1])
    >>> multi = MultiPriorDistributions([gauss, bernoulli])
    """

    def __init__(self, distributions, prior=None):
        self.prior = tolist(prior)
        super(MultiPriorDistributions,
              self).__init__(distributions, approximate=False)

    def log_likelihood_given_x(self, samples, add_prior=True, **kwargs):
        """
        Paramaters
        --------
        samples : list
           This contains 'x', which has Theano variables, and test samples,
           such as z1, z2,...,zn.

        Returns
        --------
        Theano variable, shape (n_samples,)
           log_likelihood :
             add_prior=True : log_p(zn,z'n)+log_p(zn-1|zn,z'n)+...+log_p(z2|z1)
             add_prior=False : log_p(zn-1|zn,z'n)+...+log_p(z2|z1)
        """
        all_log_likelihood = 0
        for x, sample, d in zip(samples, samples[1:], self.distributions):
            log_likelihood = d.log_likelihood_given_x([tolist(x), sample],
                                                      **kwargs)
            all_log_likelihood += log_likelihood

        if add_prior:
            for i, prior in enumerate(self.prior):
                prior_samples = samples[0][i]
                all_log_likelihood += prior.log_likelihood(prior_samples)

        return all_log_likelihood
