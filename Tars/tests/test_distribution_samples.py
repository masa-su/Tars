from unittest import TestCase

import mock
import numpy as np
from numpy.testing import (
        TestCase, run_module_suite, assert_, assert_raises, assert_equal,
        assert_warns, assert_no_warnings, assert_array_equal,
        assert_array_almost_equal)
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from theano.tensor.shared_randomstreams import RandomStreams
from theano import function, shared
from theano.tests import unittest_tools as utt
from ..distributions.distribution_samples import Bernoulli_sample, Gaussian_sample


class TestBernoulliSample(TestCase):
    def setUp(self):
        self.bernoulli_sample = Bernoulli_sample(temp=0.01, seed=1)

    @staticmethod
    def get_samples(mean, bernoulli, size):
        # get a sample from given bernoulli distribution in ndarray
        mean_sample = np.ones(size).astype("float32") * mean
        t_mean = T.fvector("mean")
        sample = bernoulli.sample(t_mean)
        f = theano.function(inputs = [t_mean], outputs=sample)
        samples = f(mean_sample)
        return samples

    def test_mean_zero(self):
        # Tests the corner case of mean == 0 for the bernoulli distribution.
        # All elements of Bernoulli_sample.sample(mean=0) should be zero.
        # ref: https://github.com/numpy/numpy/blob/master/numpy/random/tests/test_random.py
        zeros = np.zeros(1000, dtype='float')
        mean = 0
        samples = TestBernoulliSample.get_samples(mean, self.bernoulli_sample, 1000)
        self.assertTrue(np.allclose(zeros, samples))

    def test_mean_one(self):
        # Tests the corner case of mean == 1 for the bernoulli distribution.
        # All elements of Bernoulli_sample.sample(mean=1) should be one.
        # ref: https://github.com/numpy/numpy/blob/master/numpy/random/tests/test_random.py
        ones = np.ones(1000, dtype='float')
        mean = 1
        samples = TestBernoulliSample.get_samples(mean, self.bernoulli_sample, 1000)
        self.assertTrue(np.allclose(ones, samples))


class TestGaussianSample(TestCase):
    def setUp(self):
        pass

    @staticmethod
    def get_sample(mean, var, gaussian, size):
        # get a sample from given bernoulli distribution in ndarray
        mean_vector = np.ones(size).astype("float32") * mean
        var_vector = np.ones(size).astype("float32") * var
        t_mean = T.fvector("mean") # A theano symbolic variable
        t_var = T.fvector("var")
        t_sample = gaussian.sample(t_mean, t_var)
        f = theano.function(inputs = [t_mean, t_var], outputs=t_sample)
        sample = f(mean_vector, var_vector)
        return sample

    def test_gaussian(self):
        # Test that Gaussian_sample.sample generates the same result as theano and numpy
        # ref: https://github.com/Theano/Theano/blob/master/theano/tensor/tests/test_shared_randomstreams.py

        gaussian_sample = Gaussian_sample(seed=utt.fetch_seed())
        mean, var = 0, 1
        tars_sample = TestGaussianSample.get_sample(mean, var, gaussian_sample, 5)

        random = RandomStreams(utt.fetch_seed())
        fn = function([], random.normal((5,), mean, var))
        theano_sample = fn()

        rng_seed = np.random.RandomState(utt.fetch_seed()).randint(2**30)
        rng = np.random.RandomState(int(rng_seed))  # int() is for 32bit
        numpy_sample = rng.normal(mean, var, (5,))

        # As Distribution_sample.sample method performs reparametrization trick,
        # the return value is slightly different from numpy result.
        assert_array_almost_equal(tars_sample, theano_sample, decimal=7)
        assert_array_almost_equal(tars_sample, numpy_sample, decimal=7)








