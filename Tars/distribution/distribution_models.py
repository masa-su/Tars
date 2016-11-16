import numpy as np
import theano.tensor as T
import lasagne
from abc import ABCMeta, abstractmethod

from ..utils import tolist, t_repeat
from .distribution_samples import (
    Deterministic_sample,
    Bernoulli_sample,
    Categorical_sample,
    Gaussian_sample,
    GaussianConstantVar_sample,
    Laplace_sample,
    Kumaraswamy_sample,
)


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
        self.get_output_shape = lasagne.layers.get_output_shape(
            self.mean_network)

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
            raise ValueError("The length of 'x' must be same as 'given'")

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

    def sample_given_x(self, x, srng, repeat=1, **kwargs):
        """
        Paramaters
        --------
        x : list
           This contains Theano variables, which must to correspond
           to 'given'.

        srng : theano.sandbox.MRG_RandomStreams

        repeat : int or thenao variable

        Returns
        --------
        list
           This contains 'x' and sample ~ p(*|x), such as [x, sample].
        """
        if repeat != 1:
            x = [t_repeat(_x, repeat, axis=0) for _x in x]
        mean = self.fprop(x, **kwargs)
        return [x, self.sample(*tolist(mean) + [srng])]

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


class Distribution_double(Distribution):

    def __init__(self, mean_network, var_network, given):
        super(Distribution_double, self).__init__(mean_network, given)
        self.var_network = var_network
        if self.get_output_shape != lasagne.layers.get_output_shape(
                self.var_network):
            raise ValueError("The output shapes of the two networks"
                             "do not match.")

    def get_params(self):
        params = super(Distribution_double, self).get_params()
        params += self.var_network.get_params(trainable=True)
        # delete duplicated paramaters
        params = sorted(set(params), key=params.index)
        return params

    def fprop(self, x, srng=None, deterministic=False):
        mean = super(Distribution_double,
                     self).fprop(x, deterministic=deterministic)
        inputs = dict(zip(self.given, x))
        var = lasagne.layers.get_output(
            self.var_network, inputs, deterministic=deterministic)
        return mean, var


class Deterministic(Deterministic_sample, Distribution):

    def __init__(self, network, given):
        Distribution.__init__(self, network, given)


class Bernoulli(Bernoulli_sample, Distribution):

    def __init__(self, mean_network, given, temp=0.1):
        Distribution.__init__(self, mean_network, given)
        super(Bernoulli, self).__init__(temp)


class Categorical(Categorical_sample, Distribution):

    def __init__(self, mean_network, given, temp=0.1, n_dim=1):
        Distribution.__init__(self, mean_network, given)
        super(Categorical, self).__init__(temp)
        self.n_dim = n_dim
        self.k = self.get_output_shape[-1]

    def sample_given_x(self, x, srng, repeat=1, **kwargs):
        if repeat != 1:
            x = [t_repeat(_x, repeat, axis=0) for _x in x]

        # use fprop of super class
        mean = Distribution.fprop(self, x, **kwargs)
        output = self.sample(mean, srng).reshape((-1, self.n_dim * self.k))
        return [x, output]

    def fprop(self, x, *args, **kwargs):
        mean = Distribution.fprop(self, x, *args, **kwargs)
        mean = mean.reshape((-1, self.n_dim * self.k))
        return mean


class Gaussian(Gaussian_sample, Distribution_double):

    def __init__(self, mean_network, var_network, given):
        Distribution_double.__init__(self, mean_network, var_network, given)


class GaussianConstantVar(GaussianConstantVar_sample, Deterministic):

    def __init__(self, mean_network, given, var=1):
        Deterministic.__init__(self, mean_network, given)
        super(GaussianConstantVar, self).__init__(var)


class Laplace(Laplace_sample, Distribution_double):

    def __init__(self, mean_network, var_network, given):
        Distribution_double.__init__(self, mean_network, var_network, given)


class Kumaraswamy(Kumaraswamy_sample, Distribution_double):

    def __init__(self, a_network, b_network, given, stick_breaking=True):
        Distribution_double.__init__(self, a_network, b_network, given)
        self.stick_breaking = stick_breaking

    def sample_given_x(self, x, srng, repeat=1, **kwargs):
        [x, v] = super(Kumaraswamy, self).sample_given_x(x, srng,
                                                         repeat=repeat,
                                                         **kwargs)
        if self.stick_breaking:
            v = self._stick_breaking_process(v)
        return [x, v]

    def _stick_breaking_process(self, v):
        (n_batch, n_dim) = v.shape

        def segment(v, stick_segment, remaining_stick):
            stick_segment = v * remaining_stick
            remaining_stick *= (1 - v)
            return stick_segment, remaining_stick

        stick_segment = T.zeros((n_batch,))
        remaining_stick = T.ones((n_batch,))

        (stick_segments, remaining_sticks), updates\
            = theano.scan(fn=segment,
                          sequences=v.T[:-1],
                          outputs_info=[stick_segment, remaining_stick])

        return T.concatenate([stick_segments.T,
                              remaining_sticks[-1, :][:, np.newaxis]],
                             axis=1)
