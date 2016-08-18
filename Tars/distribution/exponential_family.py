import theano.tensor as T
import lasagne
from abc import ABCMeta, abstractmethod

from ..utils import gaussian_like, epsilon, tolist


# TODO: https://github.com/jych/cle/blob/master/cle/cost/__init__.py
class Distribution(object):
    __metaclass__ = ABCMeta
    """
    Paramaters
    ----------
    mean_network : lasagne.layers.Layer
       The network whose outputs express the paramater of this distribution.

    given : list
       This contains instances of lasagne.layers.InputLayer, which mean the
       conditioning variables.
       e.g. if given = [x,y], then the corresponding log-likehood is
            log p(*|x,y)
    """

    def __init__(self, mean_network, given):
        self.mean_network = mean_network
        self.given = given
        self.inputs = [x.input_var for x in given]

    def get_params(self):
        params = lasagne.layers.get_all_params(
            self.mean_network, trainable=True)
        return params

    def fprop(self, x, *args, **kwargs):
        """
        Paramaters
        ----------
        x : list
           This contains Theano variables, which must to correspond
           to 'given'.

        Returns
        -------
        mean : Theano variable
            The output of this distribution.
        """

        try:
            inputs = dict(zip(self.given, x))
        except:
            print "The length of 'x' must be same as 'given'"

        deterministic = kwargs.pop('deterministic', False)
        mean = lasagne.layers.get_output(
            self.mean_network, inputs, deterministic=deterministic)
        return mean

    def get_output_shape(self):
        """
        Returns
        -------
        tuple
          This represents the shape of the output of this distribution.
        """

        return lasagne.layers.get_output_shape(self.mean_network)

    def mean_sum_samples(self, samples):
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

    def sample_given_x(self, x, srng, **kwargs):
        """
        Paramaters
        --------
        x : list
           This contains Theano variables, which must to correspond
           to 'given'.

        srng : theano.sandbox.MRG_RandomStreams

        Returns
        --------
        list
           This contains 'x' and sample ~ p(*|x), such as [x, sample].
        """

        mean = self.fprop(x, **kwargs)
        return [x, self.sample(*tolist(mean)+[srng])]

    def sample_mean_given_x(self, x, *args, **kwargs):
        """
        Paramaters
        --------
        x : list
           This contains Theano variables, which must to correspond
           to 'given'.

        Returns
        --------
        list
           This contains 'x' and a mean value of sample ~ p(*|x).
        """

        mean = self.fprop(x, **kwargs)
        return [x, tolist(mean)[0]]

    def log_likelihood_given_x(self, samples, **kwargs):
        """
        Paramaters
        --------
        samples : list
           This contains 'x', which has Theano variables, and test sample.

        Returns
        --------
        Theano variable, shape (n_samples,)
           A log-likelihood, p(sample|x).
        """

        x, sample = samples
        mean = self.fprop(x, **kwargs)
        return self.log_likelihood(sample, *tolist(mean))

    @abstractmethod
    def sample(self):
        pass

    @abstractmethod
    def log_likelihood(self):
        pass


class Deterministic(Distribution):
    """
    Deterministic function
    p(x) = f(x)
    """

    def __init__(self, network, given):
        super(Deterministic, self).__init__(network, given)

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


class Bernoulli(Distribution):
    """
    Bernoulli distribution
    p(x) = mean^x * (1-mean)^(1-x)
    """

    def __init__(self, mean_network, given):
        super(Bernoulli, self).__init__(mean_network, given)

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

        return T.cast(T.le(srng.uniform(mean.shape), mean), mean.dtype)

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

        # for numerical stability
        mean = T.clip(mean, epsilon(), 1.0-epsilon())
        loglike = sample * T.log(mean) + (1 - sample) * T.log(1 - mean)
        return self.mean_sum_samples(loglike)


class Categorical(Bernoulli):
    """
    Categorical distribution
    p(x) = \prod mean^x
    """

    def __init__(self, mean_network, given):
        super(Categorical, self).__init__(mean_network, given)

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

        # for numerical stability
        mean = T.clip(mean, epsilon(), 1.0-epsilon())
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
            self.var_network, inputs, deterministic=deterministic)
        return mean, var

    def sample(self, mean, var, srng):
        """
        Paramaters
        ----------

        mean : Theano variable, the output of a fully connected layer (Linear)

        var : Theano variable, the output of a fully connected layer (Softplus)
        """

        eps = srng.normal(mean.shape)
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
        return self.mean_sum_samples(loglike)


class GaussianConstantVar(Deterministic):
    """
    Gaussian distribution (with a constant variance)
    p(x) = \frac{1}{\sqrt{2*\pi*var}} * exp{-\frac{{x-mean}^2}{2*var}}
    """

    def __init__(self, mean_network, given, var=1):
        super(GaussianConstantVar, self).__init__(mean_network, given)
        self.constant_var = var

    def log_likelihood(self, samples, mean):
        """
        Paramaters
        --------
        sample : Theano variable

        mean : Theano variable, the output of a fully connected layer (Linear)
        """

        loglike = gaussian_like(
            samples, mean, T.ones_like(mean)*self.constant_var)
        return self.mean_sum_samples(loglike)


class UnitGaussian(Distribution):
    """
    Standard normal gaussian distribution
    p(x) = \frac{1}{\sqrt{2*\pi}} * exp{-\frac{x^2}{2}}
    """

    def __init__(self):
        pass

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
        return T.mean(self.mean_sum_samples(loglike))


class Laplace(Gaussian):
    """
    Laplace distribution
    p(x) = \frac{1}{\sqrt{2*\phi}} * exp{-\frac{|x-mean|}{\phi}}
    """

    def __init__(self, mean_network, var_network, given):
        super(Laplace, self).__init__(mean_network, var_network, given)

    def sample(self, mean, b, srng):
        """
        Paramaters
        --------
        mean : Theano variable, the output of a fully connected layer (Linear)

        b : Theano variable, the output of a fully connected layer (Softplus)
        """

        eps = srng.uniform(mean.shape, low=-0.5, high=0.5)
        return mean - b * T.sgn(eps) * T.log(1 - 2 * abs(eps))

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
        return self.mean_sum_samples(loglike)
