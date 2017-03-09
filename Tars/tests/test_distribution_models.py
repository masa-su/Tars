from unittest import TestCase

import numpy as np
from numpy.testing import (
    assert_, assert_equal, assert_array_almost_equal
)
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from theano import function
from theano.tests import unittest_tools as utt
from ..distributions.distribution_samples import (
    BernoulliSample, GaussianSample, GumbelSample, ConcreteSample,
    CategoricalSample, LaplaceSample, KumaraswamySample,
    BetaSample, GammaSample, DirichletSample
)
from Tars.distributions.distribution_models import (
    Distribution,
    Deterministic,
    Bernoulli,
    Categorical,
    Gaussian,
    GaussianConstantVar,
    Laplace,
    Kumaraswamy,
    Gamma,
    Beta,
    Dirichlet,
)
from lasagne.layers import InputLayer,DenseLayer
from Tars.distributions import Gaussian
from lasagne.nonlinearities import rectify,linear,softplus


class TestDistributionDouble(TestCase):
    @staticmethod
    def get_samples(mean, var, size=10):
        return np.ones(size).astype("float32") * mean, np.ones(size).astype("float32") * var

    @staticmethod
    def get_sample_given_x(model, mean_sample, var_sample):
        t_mean, t_var = model.inputs
        sample = model.sample_given_x([t_mean])[-1]
        f = theano.function(inputs = [t_mean, t_var], outputs=sample)
        return f([mean_sample], [var_sample])[0]

class TestGaussian(TestCase):
    @staticmethod
    def get_model(seed=1):
        mean_layer, var_layer = InputLayer((1,None)),InputLayer((1,None))
        return Gaussian(mean_layer, var_layer, given=[mean_layer, var_layer], seed=seed)

    def test_sample_given_x_consistency(self):
        # Ensure that returned values stay the same with a fixed seed.
        seed = 1234567890
        mean, var = 0, 1
        mean_sample, var_sample = TestDistributionDouble.get_samples(mean, var)

        model = TestGaussian.get_model(seed=seed)
        model.set_seed(seed)

        actual = TestDistributionDouble.get_sample_given_x(
            model,
            mean_sample,
            var_sample
        )
        desired = [
            -1.0047914519054273e-01,
            1.2329169431414060e+00,
            -1.3303019850182538e-01,
            1.9520784223496075e+00,
            -3.5337213557864172e-01,
            1.4042016515710956e+00,
            2.1391636298546729e-01,
            -3.3813674071571591e+00,
            -9.8588173289413522e-02,
            1.9687688060042323e+00
        ]
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_sample_given_x(self):
        # The output should be the same as GaussianSample.sample when given equals to the input.
        pass

    def test_np_sample_given_x(self):
        # The output should be the same as sample_given_x
        pass
