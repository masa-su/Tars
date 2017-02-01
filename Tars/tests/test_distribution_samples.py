from unittest import TestCase

import mock
import numpy as np
import theano
import theano.tensor as T
from ..distributions.distribution_samples import Bernoulli_sample


class TestBernoulliSample(TestCase):
    def setUp(self):
        self.bernoulli_sample = Bernoulli_sample(temp=0.01, seed=1)
        pass

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

