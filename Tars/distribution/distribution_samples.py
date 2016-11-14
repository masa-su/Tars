import theano.tensor as T

from ..utils import gaussian_like, epsilon

__all__ = [
    'Deterministic_sample',
    'Bernoulli_sample',
    'Categorical_sample',
    'Gaussian_sample',
    'GaussianConstantVar_sample',
    'UnitGaussian_sample',
    'Laplace_sample',
    'Gumbel_sample',
]


class Deterministic_sample(object):
    """
    Deterministic function
    p(x) = f(x)
    """

    def sample(self, mean, *args):
        """
        Paramaters
        ----------
        mean : Theano variable, the output of a fully connected layer
               (any activation function)
        """

        return mean

    def loglikelihood(self, sample, mean):
        raise NotImplementedError


class Bernoulli_sample(object):
    """
    Bernoulli distribution
    p(x) = mean^x * (1-mean)^(1-x)
    """

    def sample(self, mean, srng):
        """
        Paramaters
        --------
        mean : Theano variable, the output of a fully connected layer (Sigmoid)
           The paramater (mean value) of this distribution.

        Returns
        -------
        Theano variable, shape (mean.shape)
           This variable is sampled from this distribution.
           i.e. sample ~ p(x|mean)
        """

        return srng.binomial(size=mean.shape, p=mean, dtype=mean.dtype)

    def log_likelihood(self, sample, mean):
        """
        Paramaters
        --------
        sample : Theano variable
           This variable means test samples which you use to estimate
           a test log-likelihood.

        mean : Theano variable, the output of a fully connected layer (Sigmoid)
           This variable is a reconstruction of test samples. This must have
           the same shape as 'sample'.

        Returns
        -------
        Theano variable, shape (n_samples,)
            A log-likelihood, which is the same meaning as a negative
            binary cross-entropy error.
        """

        # For numerical stability
        # (When we use T.clip, the calculation time becomes very slow.)
        loglike = sample * T.log(mean + epsilon()) +\
            (1 - sample) * T.log(1 - mean + epsilon())
        return mean_sum_samples(loglike)


class Categorical_sample(Bernoulli_sample):
    """
    Categorical distribution
    p(x) = \prod mean^x
    """

    def log_likelihood(self, samples, mean):
        """
        Paramaters
        --------
        sample : Theano variable
           This variable means test samples which you use to estimate
           a test log-likelihood.

        mean : Theano variable, the output of a fully connected layer (Softmax)
           This variable is a reconstruction of test samples. This must have
           the same shape as 'sample'.

        Returns
        -------
        Theano variable, shape (n_samples,)
            A log-likelihood, which is the same meaning as a negative
            categorical cross-entropy error.
        """

        # For numerical stability
        loglike = samples * T.log(mean + epsilon())
        return mean_sum_samples(loglike)


class Gaussian_sample(object):
    """
    Gaussian distribution
    p(x) = \frac{1}{\sqrt{2*\pi*var}} * exp{-\frac{{x-mean}^2}{2*var}}
    """

    def sample(self, mean, var, srng):
        """
        Paramaters
        ----------

        mean : Theano variable, the output of a fully connected layer (Linear)

        var : Theano variable, the output of a fully connected layer (Softplus)
        """

        eps = srng.normal(mean.shape, dtype=mean.dtype)
        return mean + T.sqrt(var) * eps

    def log_likelihood(self, samples, mean, var):
        """
        Paramaters
        --------
        sample : Theano variable

        mean : Theano variable, the output of a fully connected layer (Linear)

        var : Theano variable, the output of a fully connected layer (Softplus)
        """

        loglike = gaussian_like(samples, mean, var)
        return mean_sum_samples(loglike)


class GaussianConstantVar_sample(Deterministic_sample):
    """
    Gaussian distribution (with a constant variance)
    p(x) = \frac{1}{\sqrt{2*\pi*var}} * exp{-\frac{{x-mean}^2}{2*var}}
    """

    def __init__(self, var=1):
        self.constant_var = var

    def log_likelihood(self, samples, mean):
        """
        Paramaters
        --------
        sample : Theano variable

        mean : Theano variable, the output of a fully connected layer (Linear)
        """

        loglike = gaussian_like(
            samples, mean, T.ones_like(mean) * self.constant_var)
        return mean_sum_samples(loglike)


class UnitGaussian_sample(object):
    """
    Standard normal gaussian distribution
    p(x) = \frac{1}{\sqrt{2*\pi}} * exp{-\frac{x^2}{2}}
    """

    def sample(self, shape, srng):
        """
        Paramaters
        --------
        shape : tuple
           sets a shape of the output sample
        """

        return srng.normal(shape)

    def log_likelihood(self, samples):
        """
        Paramaters
        --------
        sample : Theano variable
        """

        loglike = gaussian_like(samples,
                                T.zeros_like(samples), T.ones_like(samples))
        return mean_sum_samples(loglike)


class Laplace_sample(object):
    """
    Laplace distribution
    p(x) = \frac{1}{\sqrt{2*\phi}} * exp{-\frac{|x-mean|}{\phi}}
    """

    def sample(self, mean, b, srng):
        """
        Paramaters
        --------
        mean : Theano variable, the output of a fully connected layer (Linear)

        b : Theano variable, the output of a fully connected layer (Softplus)
        """

        U = srng.uniform(mean.shape, low=-0.5, high=0.5, dtype=mean.dtype)
        return mean - b * T.sgn(U) * T.log(1 - 2 * abs(U))

    def log_likelihood(self, samples, mean, b):
        """
        Paramaters
        --------
        sample : Theano variable

        mean : Theano variable, the output of a fully connected layer (Linear)

        b : Theano variable, the output of a fully connected layer (Softplus)
        """

        # for numerical stability
        b += epsilon()
        loglike = -abs(samples - mean) / b - T.log(b) - T.log(2)
        return mean_sum_samples(loglike)


class Gumbel_sample(object):
    """
    Gumbel distribution
    """

    def sample(self, mu, beta, srng):
        U = srng.uniform(mu.shape, low=0, high=1, dtype=mu.dtype)
        return mu - beta * T.log(T.log(U + epsilon()))

    def log_likelihood(self, samples, mu, beta):
        """
        Paramaters
        --------
        sample : Theano variable

        mu : Theano variable, the output of a fully connected layer (Linear)

        beta : Theano variable, the output of a fully connected layer
        (Softplus)
        """

        z = (samples - mu) / (beta + epsilon())
        loglike = -T.log(beta) - (z + T.exp(-z))
        return mean_sum_samples(loglike)


def mean_sum_samples(samples):
    n_dim = samples.ndim
    if n_dim == 4:
        return T.mean(T.sum(T.sum(samples, axis=2), axis=2), axis=1)
    elif n_dim == 3:
        return T.sum(T.sum(samples, axis=-1), axis=-1)
    elif n_dim == 2:
        return T.sum(samples, axis=-1)
    else:
        raise ValueError("The dim of samples must be any of 2, 3, or 4,"
                         "got dim %s." % n_dim)
