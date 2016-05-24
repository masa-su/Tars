import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from util import gaussian_like, tolist
import lasagne


class Distribution(object):

    def __init__(self, mean_network, given):
        self.mean_network = mean_network
        self.given = given
        self.inputs = [x.input_var for x in given]

    def get_params(self):
        params = lasagne.layers.get_all_params(
            self.mean_network, trainable=True)
        return params

    def fprop(self, x, srng=None, deterministic=False):
        inputs = dict(zip(self.given, x))
        mean = lasagne.layers.get_output(
            self.mean_network, inputs, deterministic=deterministic)
        return mean

    def mean_sum_samples(self, samples):
        n_dim = samples.ndim
        if n_dim == 2:
            return T.sum(samples, axis=1)
        elif n_dim == 4:
            return T.mean(T.sum(T.sum(samples, axis=2), axis=2), axis=1)


class Bernoulli(Distribution):
    """
    Bernoulli distribution
    p(x) = mean^x * (1-mean)^(1-x)
    """

    def __init__(self, mean_network, given):
        super(Bernoulli, self).__init__(mean_network, given)

    def fprop(self, x, srng=None, deterministic=False):
        mean = super(Bernoulli, self).fprop(x, deterministic)
        return mean

    def sample(self, mean, srng):
        return T.cast(T.le(srng.uniform(mean.shape), mean), mean.dtype)

    def log_likelihood(self, samples, mean):
        loglike = samples * T.log(mean) + (1 - samples) * T.log(1 - mean)
        return self.mean_sum_samples(loglike)

    def sample_given_x(self, x, srng, deterministic=False):
        """
        inputs : x
        outputs : [x,z]
        """
        mean = self.fprop(x, deterministic=deterministic)
        return [x, self.sample(mean, srng)]

    def sample_mean_given_x(self, x, srng=None, deterministic=False):
        """
        inputs : x
        outputs : [x,z]
        """
        mean = self.fprop(x, deterministic=deterministic)
        return [x, mean]

    def log_likelihood_given_x(self, samples, deterministic=False):
        x, sample = samples
        mean = self.fprop(x, deterministic=deterministic)
        return self.log_likelihood(sample, mean)


class Categorical(Bernoulli):
    """
    Categorical distribution
    p(x) = \prod mean^x
    """

    def __init__(self, mean_network, given):
        super(Categorical, self).__init__(mean_network, given)

    def log_likelihood(self, samples, mean):
        loglike = samples * T.log(mean)
        return self.mean_sum_samples(loglike)


class Gaussian(Distribution):
    """
    Gaussian distribution
    p(x) = \frac{1}{\sqrt{2*\pi*var}} * exp{-\frac{{x-mean}^2}{2*var}}
    """

    def __init__(self, mean_network, var_network, given):
        super(Gaussian, self).__init__(mean_network, given)
        self.var_network = var_network

    def get_params(self):
        params = super(Gaussian, self).get_params()
        params += self.var_network.get_params(trainable=True)
        return params

    def fprop(self, x, srng=None, deterministic=False):
        mean = super(Gaussian, self).fprop(x, deterministic)
        inputs = dict(zip(self.given, x))
        var = lasagne.layers.get_output(
            self.var_network, inputs, deterministic=deterministic)  # simga**2
        return mean, var

    def sample(self, mean, var, srng):
        eps = srng.normal(mean.shape)
        return mean + T.sqrt(var) * eps

    def log_likelihood(self, samples, mean, var):
        loglike = gaussian_like(samples, mean, var)
        return self.mean_sum_samples(loglike)

    def sample_given_x(self, x, srng, deterministic=False):
        """
        inputs : x
        outputs : [x,z]
        """
        mean, var = self.fprop(x, deterministic=deterministic)
        return [x, self.sample(mean, var, srng)]

    def sample_mean_given_x(self, x, srng=None, deterministic=False):
        """
        inputs : x
        outputs : [x,z]
        """
        mean, _ = self.fprop(x, deterministic=deterministic)
        return [x, mean]

    def log_likelihood_given_x(self, samples, deterministic=False):
        x, sample = samples
        mean, var = self.fprop(x, deterministic=deterministic)
        return self.log_likelihood(sample, mean, var)


class Gaussian_nonvar(Bernoulli):
    """
    Gaussian distribution (no variance)
    p(x) = \frac{1}{\sqrt{2*\pi}} * exp{-\frac{{x-mean}^2}{2}}
    """

    def __init__(self, mean_network, given):
        super(Gaussian_nonvar, self).__init__(mean_network, given)

    def log_likelihood(self, samples, mean):
        loglike = gaussian_like(samples, mean, T.ones_like(mean))
        return self.mean_sum_samples(loglike)


class UnitGaussian(Distribution):
    """
    Standard normal gaussian distribution
    p(x) = \frac{1}{\sqrt{2*\pi}} * exp{-\frac{x^2}{2}}
    """

    def __init__(self):
        pass

    def sample(self, shape, srng):
        return srng.normal(shape)

    def log_likelihood(self, samples):
        loglike = gaussian_like(samples,
                                T.zeros_like(samples), T.ones_like(samples))
        return T.mean(self.mean_sum_samples(loglike))


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
            sample = d.sample_given_x(samples[i], srng, deterministic=deterministic)
            samples.append(sample[-1])
        return samples

    def fprop(self, x, srng, deterministic=False):
        """
        inputs : x
        outputs : mean
        """
        samples = self.__sampling(x, srng, deterministic)
        sample = self.distributions[-1].fprop(samples[-1:], deterministic=deterministic)
        return sample

    def sample_given_x(self, x, srng, deterministic=False):
        """
        inputs : x
        outputs : [x,z1,...,zn]
        """
        samples = self.__sampling(x, srng, deterministic)        
        samples += self.distributions[-1].sample_given_x(samples[-1:], srng, deterministic=deterministic)[-1:]
        return samples

    def sample_mean_given_x(self, x, srng, deterministic=False):
        """
        inputs : x
        outputs : [x,z1,...,zn]
        """
        mean = self.__sampling(x, srng, deterministic)
        mean += self.distributions[-1].sample_mean_given_x(mean[-1:], deterministic=deterministic)[-1:]
        return mean

    def log_likelihood_given_x(self, samples, deterministic=False):
        """
        inputs : [[x,y,...],z1,z2,...,zn] or [[zn,y,...],zn-1,...,x]
        log_likelihood (q) : [q(z1|[x,y,...]),...,q(zn|zn-1)]
        log_likelihood (p) : [p(zn-1|[zn,y,...]),...,p(x|z1)]
        """
        all_log_likelihood = 0
        for x, sample, d in zip(samples, samples[1:], self.distributions):
            log_likelihood = d.log_likelihood_given_x([tolist(x),sample])
            all_log_likelihood += log_likelihood
        return all_log_likelihood
