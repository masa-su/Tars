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
        self.output_dim = 0
        for d in self.distributions:
            if self.distributions[0].given != d.given:
                raise ValueError("Every distribution must have same"
                                 "given variables")
            self.output_dim += d.get_output_shape()[-1]
            
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

        if sample.shape[-1] != self.output_dim:
            raise ValueError("The dimension of test sample must be same as"
                             "output_dim.")

        loglikes = 0
        start = 0
        for d in enumerate(self.distributions):
            shape = d.get_output_shape()[-1]
            loglikes += d.log_likelihood_given_x(
                [x, sample[:, start:start+shape]],
                **kwargs)
            start += shape
        return loglikes


# TODO: conplicated conditional version
class Multilayer(object):
    """
    Multiple layer distribution
    p(x|z) = p(x|z1)p(z1|z2)...p(zn-1|zn)
    q(z|x) = q(zn|zn-1)...q(z2|z1)q(z1|x)
    """

    def __init__(self, distributions):
        self.distributions = distributions
        self.inputs = self.distributions[0].inputs

    def get_params(self):
        params = []
        for d in self.distributions:
            params += d.get_params()
        return params

    def __sampling(self, x, srng, deterministic):
        """
        inputs : x
        outputs : [x,z1,...,zn-1]
        """
        samples = [x]
        for i, d in enumerate(self.distributions[:-1]):
            sample = d.sample_given_x(
                samples[i], srng, deterministic=deterministic)
            samples.append(sample[-1])
        return samples

    def fprop(self, x, srng, deterministic=False):
        """
        inputs : x
        outputs : mean
        """
        samples = self.__sampling(x, srng, deterministic)
        sample = self.distributions[-1].fprop(
            tolist(samples[-1]), deterministic=deterministic)
        return sample

    def sample_given_x(self, x, srng, deterministic=False):
        """
        inputs : x
        outputs : [x,z1,...,zn]
        """
        samples = self.__sampling(x, srng, deterministic)
        samples += self.distributions[-1].sample_given_x(
            tolist(samples[-1]), srng, deterministic=deterministic)[-1:]
        return samples

    def sample_mean_given_x(self, x, srng, deterministic=False):
        """
        inputs : x
        outputs : [x,z1,...,zn]
        """
        mean = self.__sampling(x, srng, deterministic)
        mean += self.distributions[-1].sample_mean_given_x(
            tolist(mean[-1]), deterministic=deterministic)[-1:]
        return mean

    def log_likelihood_given_x(self, samples, deterministic=False):
        """
        inputs : [[x,y,...],z1,z2,...,zn] or [[zn,y,...],zn-1,...,x]
        outputs :
           log_likelihood (q) : [q(z1|[x,y,...]),...,q(zn|zn-1)]
           log_likelihood (p) : [p(zn-1|[zn,y,...]),...,p(x|z1)]
        """
        all_log_likelihood = 0
        for x, sample, d in zip(samples, samples[1:], self.distributions):
            log_likelihood = d.log_likelihood_given_x([tolist(x), sample])
            all_log_likelihood += log_likelihood
        return all_log_likelihood
