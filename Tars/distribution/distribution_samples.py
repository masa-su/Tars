import math
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from ..utils import epsilon

__all__ = [
    'Deterministic_sample',
    'Bernoulli_sample',
    'Categorical_sample',
    'UnitBernoulli_sample',
    'UnitCategorical_sample',
    'Gaussian_sample',
    'UnitGaussian_sample',
    'Laplace_sample',
    'Gumbel_sample',
    'Concrete_sample',
    'Kumaraswamy_sample',
    'UnitBeta_sample',
]


class Distribution_sample(object):

    def __init__(self, seed=1, **kwargs):
        self.srng = RandomStreams(seed)

    def set_seed(self, seed=1):
        self.srng = RandomStreams(seed)


class Deterministic_sample(object):
    """
    Deterministic function
    p(x) = f(x)
    """

    def __init__(self, **kwargs):
        pass

    def sample(self, mean, *args):
        """
        Paramaters
        ----------
        mean : Theano variable, the output of a fully connected layer
               (any activation function)
        """

        return mean

    def log_likelihood(self, sample, mean):
        raise NotImplementedError


class Gumbel_sample(Distribution_sample):
    """
    Gumbel distribution
    """

    def sample(self, mu, beta):
        U = self.srng.uniform(mu.shape,
                              low=0, high=1, dtype=mu.dtype)
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

    def __init__(self, temp=0.1, seed=1):
        super(Concrete_sample, self).__init__(seed=seed)
        self.temp = temp

    def sample(self, mean):
        """
        Paramaters
        --------
        sample : Theano variable

        mean : Theano variable, the output of a fully connected layer
        (sigmoid or softmax)
        """

        if self.temp != 0:
            output = super(Concrete_sample, self).sample(T.zeros_like(mean),
                                                         T.ones_like(mean))
            output += T.log(mean + epsilon())

            if output.ndim == 1 or output.ndim == 2:
                return T.nnet.softmax(output / self.temp)
            elif output.ndim == 3:
                _shape = output.shape
                output = output.reshape((_shape[0] * _shape[1], _shape[2]))
                return T.nnet.softmax(output / self.temp).reshape(_shape)

            raise ValueError('Input must be 1-d, 2-d or 3-d tensor. Got %s' %
                             output.ndim)

        raise NotImplementedError

    def log_likelihood(self):
        raise NotImplementedError


class Bernoulli_sample(Gumbel_sample):
    """
    Bernoulli distribution
    p(x) = mean^x * (1-mean)^(1-x)
    """

    def __init__(self, temp=0.1, seed=1):
        super(Bernoulli_sample, self).__init__(seed=seed)
        self.temp = temp

    def sample(self, mean):
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

        if self.temp != 0:
            z1 = super(Bernoulli_sample, self).sample(T.zeros_like(mean),
                                                      T.ones_like(mean))
            z0 = super(Bernoulli_sample, self).sample(T.zeros_like(mean),
                                                      T.ones_like(mean))
            z1 += T.log(mean + epsilon())
            z0 += T.log(1 - mean + epsilon())

            return T.nnet.sigmoid((z1 - z0) / self.temp)

        raise NotImplementedError

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


class Categorical_sample(Concrete_sample):
    """
    Categorical distribution
    p(x) = \prod mean^x
    """

    def __init__(self, temp=0.1, seed=1):
        super(Categorical_sample, self).__init__(temp=temp, seed=seed)

    def sample(self, mean, onehot=True, flatten=True):
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

        if mean.ndim == 1 or mean.ndim == 2:
            output = super(Categorical_sample, self).sample(mean)
            if not onehot:
                output = T.argmax(output, axis=-1)
            return output

        elif mean.ndim == 3:
            _shape = mean.shape
            mean = mean.reshape((_shape[0] * _shape[1], _shape[2]))
            output = super(Categorical_sample, self).sample(
                mean).reshape(_shape)
            if not onehot:
                output = T.argmax(output, axis=-1)
            if flatten:
                output = T.flatten(output, outdim=2)
            return output

        raise ValueError('Wrong the dimention of input.')

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


class UnitBernoulli_sample(Bernoulli_sample):
    """
    Unit bernoulli distribution
    """

    def sample(self, shape):
        return super(UnitBernoulli_sample,
                     self).sample(T.ones(shape) * 0.5)

    def log_likelihood(self, samples):
        return super(UnitBernoulli_sample,
                     self).log_likelihood(samples,
                                          T.ones_like(samples) * 0.5)


class UnitCategorical_sample(Categorical_sample):
    """
    Unit Categorical distribution
    """

    def __init__(self, k=1, seed=1):
        super(UnitCategorical_sample, self).__init__(seed=seed)
        self.k = k

    def sample(self, shape):
        if self.k == shape[-1]:
            return super(UnitCategorical_sample,
                         self).sample(T.ones(shape) / self.k)

        raise ValueError("self.k and shape don't match.")

    def log_likelihood(self, samples):
        return super(UnitCategorical_sample,
                     self).log_likelihood(samples,
                                          T.ones_like(samples) / self.k)


class Gaussian_sample(Distribution_sample):
    """
    Gaussian distribution
    p(x) = \frac{1}{\sqrt{2*\pi*var}} * exp{-\frac{{x-mean}^2}{2*var}}
    """

    def sample(self, mean, var):
        """
        Paramaters
        ----------

        mean : Theano variable, the output of a fully connected layer (Linear)

        var : Theano variable, the output of a fully connected layer (Softplus)
        """

        eps = self.srng.normal(mean.shape, dtype=mean.dtype)
        return mean + T.sqrt(var) * eps

    def log_likelihood(self, samples, mean, var):
        """
        Paramaters
        --------
        sample : Theano variable

        mean : Theano variable, the output of a fully connected layer (Linear)

        var : Theano variable, the output of a fully connected layer (Softplus)
        """

        loglike = self._gaussian_like(samples, mean, var)
        return mean_sum_samples(loglike)

    def _gaussian_like(self, x, mean, var):
        c = - 0.5 * math.log(2 * math.pi)
        _var = var + epsilon()  # avoid NaN
        return c - T.log(_var) / 2 - (x - mean)**2 / (2 * _var)


class UnitGaussian_sample(Gaussian_sample):
    """
    Standard normal gaussian distribution
    p(x) = \frac{1}{\sqrt{2*\pi}} * exp{-\frac{x^2}{2}}
    """

    def sample(self, shape):
        """
        Paramaters
        --------
        shape : tuple
           sets a shape of the output sample
        """

        return self.srng.normal(shape)

    def log_likelihood(self, samples):
        return super(UnitGaussian_sample,
                     self).log_likelihood(samples,
                                          T.zeros_like(samples),
                                          T.ones_like(samples))


class Laplace_sample(Distribution_sample):
    """
    Laplace distribution
    p(x) = \frac{1}{\sqrt{2*\phi}} * exp{-\frac{|x-mean|}{\phi}}
    """

    def sample(self, mean, b):
        """
        Paramaters
        --------
        mean : Theano variable, the output of a fully connected layer (Linear)

        b : Theano variable, the output of a fully connected layer (Softplus)
        """

        U = self.srng.uniform(mean.shape,
                              low=-0.5, high=0.5,
                              dtype=mean.dtype)
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


class Kumaraswamy_sample(Distribution_sample):
    """
    [Naelisnick+ 2016] Deep Generative Models with Stick-Breaking Priors
    Kumaraswamy distribution
    p(x) = a*b*x^(a-1)(1-x^a)^(b-1)
    """

    def sample(self, a, b):
        """
        Paramaters
        --------
        a : Theano variable, the output of a fully connected layer (Softplus)
        b : Theano variable, the output of a fully connected layer (Softplus)
        """

        eps = self.srng.uniform(a.shape,
                                low=epsilon(),
                                high=1 - epsilon(),
                                dtype=a.dtype)
        return (1 - eps**(1. / b))**(1. / a)

    def log_likelihood(self, samples, a, b):
        """
        Paramaters
        --------
        sample : Theano variable
        a : Theano variable, the output of a fully connected layer (Softplus)
        b : Theano variable, the output of a fully connected layer (Softplus)
        """

        loglike = T.log(a * b + epsilon())\
            + (a - 1) * T.log(samples + epsilon())\
            + (b - 1) * T.log(1 - samples**a + epsilon())
        return mean_sum_samples(loglike)


class Gamma_sample(Distribution_sample):
    """
    Gamma distribution
    (beta^alpha)/gamma * x^(alpha-1) * exp^(-beta*x)

    [Naesseth+ 2016]
    Rejection Sampling Variational Inference
    """

    def __init__(self, seed=1):
        super(Gamma_sample, self).__init__(seed=seed)

    def sample(self, alpha, beta):
        _shape = alpha.shape
        alpha = alpha.flatten()
        output_sample = -T.ones_like(alpha, dtype=alpha.dtype)
        index = T.arange(output_sample.shape[0])

        # We don't use theano.scan in order to avoid to use updates.
        output_sample = self._rejection_sampling(output_sample, alpha, index)
        output_sample = self._rejection_sampling(output_sample, alpha, index)
        output_sample = self._rejection_sampling(output_sample, alpha, index)
        output_sample = self._rejection_sampling(output_sample, alpha, index)
        output_sample = self._rejection_sampling(output_sample, alpha, index)
        output_sample = T.clip(output_sample, 0, output_sample)

        return output_sample.reshape(_shape) / beta

    def log_likelihood(self, samples, alpha, beta):
        output = alpha * T.log(beta + epsilon()) - T.gammaln(alpha)
        output += (alpha - 1) * T.log(samples + epsilon())
        output += -beta * samples
        return mean_sum_samples(output)

    def pdf(self, samples, alpha, beta):
        output = (beta**alpha) / T.gamma(alpha)
        output *= samples**(alpha - 1)
        output *= T.exp(-beta * samples)
        return output

    def _gauss_pdf(self, sample):
        output = 1 / T.sqrt(2 * math.pi)
        output *= T.exp(-sample**2 / 2)
        return output

    def _h(self, alpha, eps):
        output = alpha - 1 / 3.
        output *= (1 + eps / T.sqrt(alpha * 9 - 3))**3
        return output

    def _rejection_sampling(self, output_z, alpha, idx):
        idx = idx[T.eq(-1, output_z).nonzero()]
        under_one_idx = idx[T.gt(1, alpha[idx]).nonzero()]
        added_alpha = T.inc_subtensor(alpha[under_one_idx], 1)

        eps = self.srng.normal(idx.shape, dtype=alpha.dtype)
        U = self.srng.uniform(idx.shape,
                              low=epsilon(),
                              high=1 - epsilon(),
                              dtype=alpha.dtype)
        z = self._h(added_alpha[idx], eps)

        _idx = T.lt(U, self.pdf(z, added_alpha[idx], T.ones_like(
            added_alpha[idx])) / self._gauss_pdf(z)).nonzero()
        idx = idx[_idx]
        output_z = T.set_subtensor(output_z[idx], z[_idx])

        _under_one_idx = T.arange(idx.shape[0])[T.gt(1, alpha[idx]).nonzero()]
        under_one_idx = idx[_under_one_idx]
        output_z = T.set_subtensor(output_z[under_one_idx],
                                   (U[_under_one_idx] **
                                    (1 / (alpha[under_one_idx]))) *
                                   output_z[under_one_idx])

        return output_z


class UnitGamma_sample(Gamma_sample):

    def sample(self, shape):
        """
        Paramaters
        --------
        shape : tuple
           sets a shape of the output sample
        """

        return super(UnitGamma_sample,
                     self).sample(T.ones(shape),
                                  T.ones(shape)*0.01)

    def log_likelihood(self, samples):
        return super(UnitGamma_sample,
                     self).log_likelihood(samples,
                                          T.ones_like(samples),
                                          T.ones_like(samples)*0.01)


class UnitBeta_sample(Distribution_sample):
    """
    Unit Beta distribution
    """

    def __init__(self, alpha=1., beta=5., seed=1):
        super(UnitBeta_sample, self).__init__(seed=seed)
        self.alpha = alpha
        self.beta = beta

    def sample(self, shape):
        raise NotImplementedError

    def log_likelihood(self, samples):
        output = -T.log(self._beta_func(self.alpha, self.beta) + epsilon())
        output += (self.alpha - 1) * samples
        output += (self.beta - 1) * (1 - samples)
        return output

    def _beta_func(self, a, b):
        return T.exp(T.gammaln(a) + T.gammaln(b) - T.gammaln(a + b))


def mean_sum_samples(samples):
    n_dim = samples.ndim
    if n_dim == 4:
        return T.mean(T.sum(T.sum(samples, axis=2), axis=2), axis=1)
    elif n_dim == 3:
        return T.sum(T.sum(samples, axis=-1), axis=-1)
    elif n_dim == 2:
        return T.sum(samples, axis=-1)
    raise ValueError("The dim of samples must be any of 2, 3, or 4,"
                     "got dim %s." % n_dim)
