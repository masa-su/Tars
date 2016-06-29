import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import lasagne

from ..util import gaussian_like, tolist, epsilon


# TODO: https://github.com/jych/cle/blob/master/cle/cost/__init__.py
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

    def get_output_shape(self):
        return self.mean_network.get_output_shape_for(self.inputs)

    def mean_sum_samples(self, samples):
        n_dim = samples.ndim
        if n_dim == 4: #TODO:fix 
            return T.mean(T.sum(T.sum(samples, axis=2), axis=2), axis=1)
        else:
            return T.sum(samples, axis=-1)

class Deterministic(Distribution):
    """
    Deterministic function
    p(x) = f(x)
    """

    def __init__(self, network, given):
        super(Deterministic, self).__init__(network, given)

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
        mean = T.clip(mean, epsilon(), 1.0-epsilon()) # for numerical stability
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
        """
        inputs : [x,sample]
        outputs : p(sample|x)
        """
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
        mean = T.clip(mean, epsilon(), 1.0-epsilon()) # for numerical stability
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
        # TODO: fix duplicated paramaters
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

class BivariateGauss(Gaussian):

    def __init__(self, mean_network, var_network, corr_network, given):
        super(BivariateGauss, self).__init__(mean_network, var_network, corr_network, given)

    def get_params(self):
        params = super(BivariateGauss, self).get_params()
        params += self.corr_network.get_params(trainable=True)
        return params

    def fprop(self, x, srng=None, deterministic=False):
        mean, var = super(BivariateGauss, self).fprop(x, deterministic)
        inputs = dict(zip(self.given, x))
        corr = lasagne.layers.get_output(
            self.corr_network, inputs, deterministic=deterministic)
        return mean, var, corr

    def sample(self, mean, var, corr, srng):
        # Cholesky
        L = T.zeros((mean.shape[1],mean.shape[1]))
        L[0,0] = T.sqrt(var[:,0][np.newaxis])
        L[1,0] = corr/L[0,0]
        L[1,1] = T.sqrt(var[:,1][np.newaxis] - L[1,0]**2)

        eps = srng.normal(mean.shape)
        return mean + T.dot(eps,L)

    def log_likelihood(self, samples, mean, var):
        mean_0 = mean[:, 0].reshape((-1, 1))
        mean_1 = mean[:, 1].reshape((-1, 1))

        var_0 = T.clip(var[:, 0].reshape((-1, 1)), epsilon(), np.inf)
        var_1 = T.clip(var[:, 1].reshape((-1, 1)), epsilon(), np.inf)

        samples_0 = samples[:, 0].reshape((-1, 1))
        samples_1 = samples[:, 1].reshape((-1, 1))
        corr = T.clip(corr.reshape((-1, 1)), epsilon(), 1.0-epsilon())

        inner1 =  ((0.5*T.log(1-corr**2)) +
                   0.5 * T.log(var_0) + 0.5 * T.log(var_1) + T.log(2 * np.pi))

        z = ((samples_0 - mean_0))**2 / var_0 + ((samples_1 - mean_1))**2 / var_1 \
             - (2. * (corr * (samples_0 - mean_0) * (samples_1 - mean_1)) / (T.sqrt(var_0) * T.sqrt(var_1)))

        inner2 = 0.5 * (1. / (1. - corr**2))
        loglike = inner1 + (inner2 * z)

        return self.mean_sum_samples(loglike)

    def sample_given_x(self, x, srng, deterministic=False):
        """
        inputs : x
        outputs : [x,z]
        """
        mean, var, corr = self.fprop(x, deterministic=deterministic)
        return [x, self.sample(mean, var, corr, srng)]

    def sample_mean_given_x(self, x, srng=None, deterministic=False):
        """
        inputs : x
        outputs : [x,z]
        """
        mean, _, _ = self.fprop(x, deterministic=deterministic)
        return [x, mean]

    def log_likelihood_given_x(self, samples, deterministic=False):
        x, sample = samples
        mean, var, corr = self.fprop(x, deterministic=deterministic)
        return self.log_likelihood(sample, mean, var, corr)


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

class Laplace(Gaussian):
    """
    Laplace distribution
    p(x) = \frac{1}{\sqrt{2*\phi}} * exp{-\frac{|x-mean|}{\phi}}
    """

    def __init__(self, mean_network, var_network, given):
        super(Laplace, self).__init__(mean_network, var_network, given)

    def sample(self, mean, b, srng):
        eps = srng.uniform(mean.shape, low=-0.5, high=0.5)
        return mean -b * T.sgn(eps) * T.log(1 - 2 * abs(eps))

    def log_likelihood(self, samples, mean, b):
        b += epsilon() # for numerical stability
        loglike = -abs(samples - mean) / b - T.log(b) - T.log(2)
        return self.mean_sum_samples(loglike)
