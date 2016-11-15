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
    'Concrete_sample',
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

    def __init__(self, temp=0.1):
        self.temp = temp
        self.concrete = Concrete_sample(temp)

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

        mean = T.stack([mean, 1-mean], axis=-1)
        output = self.concrete.sample(mean, srng)

        if output.ndim == 2:
            return output[:, 0]
        elif output.ndim == 3:
            return output[:, :, 0]
        else:
            raise ValueError('Input must be 1-d or 2-d tensor. Got %s' %
                             mean.ndim)

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


class Categorical_sample(object):
    """
    Categorical distribution
    p(x) = \prod mean^x
    """

    def __init__(self, temp=0.1, n_dim=1):
        self.temp = temp
        self.concrete = Concrete_sample(temp)
        self.n_dim = n_dim

    def sample(self, mean, srng, onehot=True, flatten=True):
        """
        Paramaters
        --------
        mean : Theano variable, the output of a fully connected layer (softmax)
           The paramater (mean value) of this distribution.

        Returns
        -------
        Theano variable, shape (mean.shape)
           This variable is sampled from this distribution.
           i.e. sample ~ p(x|mean)
        """

        if self.n_dim == 1 and (mean.ndim == 1 or mean.ndim == 2):
            output = self.concrete.sample(mean, srng)
            if not onehot:
                output = T.argmax(output, axis=-1)
            return output

        elif self.n_dim >= 1 and mean.ndim == 3:
            if mean.shape[1].eval() == self.n_dim:
                _shape = mean.shape
                mean = mean.reshape((_shape[0]*_shape[1], _shape[2]))
                output = self.concrete.sample(mean, srng).reshape(_shape)
                if not onehot:
                    output = T.argmax(output, axis=-1)
                if flatten:
                    output = T.flatten(output)
                return output

            raise ValueError('mean.shape[1] is incongruous with n_dim,'
                             'got mean.shape[1] = %s and n_dim = %s'
                             % (mean.shape[1].eval(), mean.ndim))

        raise ValueError('Wrong input or n_dim.')

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


class GaussianConstantVar_sample(Gaussian_sample):
    """
    Gaussian distribution (with a constant variance)
    p(x) = \frac{1}{\sqrt{2*\pi*var}} * exp{-\frac{{x-mean}^2}{2*var}}
    """

    def __init__(self, var=1):
        self.constant_var = var

    def sample(self, samples, mean):
        return super(GaussianConstantVar_sample,
                     self).sample(samples, mean,
                                  T.ones_like(samples) * self.constant_var)

    def log_likelihood(self, samples, mean):
        return super(GaussianConstantVar_sample,
                     self).log_likelihood(samples, mean,
                                          T.ones_like(samples)
                                          * self.constant_var)


class UnitGaussian_sample(Gaussian_sample):
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
        return super(UnitGaussian_sample,
                     self).log_likelihood(samples,
                                          T.zeros_like(samples),
                                          T.ones_like(samples))


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
        return mu - beta * T.log(-T.log(U + epsilon()) + epsilon())

    def log_likelihood(self, samples, mu, beta):
        """
        Paramaters
        --------
        sample : Theano variable

        mu : Theano variable, the output of a fully connected layer (Linear)

        beta : Theano variable, the output of a fully connected layer
        (Softplus)
        """

        # for numerical stability
        beta += epsilon()
        z = (samples - mu) / beta
        loglike = -T.log(beta) - (z + T.exp(-z))
        return mean_sum_samples(loglike)


class Concrete_sample(Gumbel_sample):
    """
    Concrete distribution (Gumbel-softmax)
        https://arxiv.org/abs/1611.01144
        https://arxiv.org/abs/1611.00712
    """

    def __init__(self, temp=0.1):
        self.temp = temp

    def sample(self, mean, srng):
        """
        Paramaters
        --------
        sample : Theano variable

        mean : Theano variable, the output of a fully connected layer
        (sigmoid or softmax)
        """

        if self.temp != 0:
            output = super(Concrete_sample, self).sample(T.zeros_like(mean),
                                                         T.ones_like(mean),
                                                         srng)
            output += T.log(mean + epsilon())

            if output.ndim == 1 or output.ndim == 2:
                return T.nnet.softmax(output / self.temp)
            elif output.ndim == 3:
                _shape = output.shape
                output = output.reshape((_shape[0]*_shape[1], _shape[2]))
                return T.nnet.softmax(output / self.temp).reshape(_shape)

            raise ValueError('Input must be 1-d, 2-d or 3-d tensor. Got %s' %
                             output.ndim)

        raise NotImplementedError

    def log_likelihood(self):
        raise NotImplementedError


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
