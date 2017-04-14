from unittest import TestCase

import scipy
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


def get_sample(mean, distribution_sample, size, t_mean=None):
    if t_mean is None:
        t_mean = T.fvector("mean")
    # get a sample from given distribution in ndarray
    mean_vector = np.ones(size).astype("float32") * mean
    t_sample = distribution_sample.sample(t_mean)
    f = theano.function(inputs=[t_mean], outputs=t_sample)

    if t_mean.ndim == 1:
        return f(mean_vector)
    elif t_mean.ndim == 2:
        return f([mean_vector])[0]


def get_sample_double(mean, var, distribution_sample_double, size, t_mean=None, t_var=None):
    if t_mean is None or t_var is None:
        t_mean = T.fvector("mean")
        t_var = T.fvector("var")
    # get a sample from given gaussian distribution in ndarray
    mean_vector = np.ones(size).astype("float32") * mean
    var_vector = np.ones(size).astype("float32") * var

    t_sample = distribution_sample_double.sample(t_mean, t_var)
    f = theano.function(inputs=[t_mean, t_var], outputs=t_sample)

    if t_mean.ndim == 1:
        return f(mean_vector, var_vector)
    elif t_mean.ndim == 2:
        return f([mean_vector], [var_vector])[0]


class TestGumbelSample(TestCase):
    def setUp(self):
        self.seed = 1234567890

    def test_consistency(self):
        # Ensure that returned values stay the same when setting a fixed seed.
        mu, beta = 0, 1
        gumbel_sample = GumbelSample(temp=0.01, seed=self.seed)
        actual = get_sample_double(mu, beta, gumbel_sample, 5)
        desired = [
            1.7462246417999268,
            0.2874840497970581,
            0.9974965453147888,
            0.3290553689002991,
            -1.0215276479721069
        ]
        assert_array_almost_equal(actual, desired, decimal=6)

    def test_mean_zero(self):
        mu, beta = 0, 0
        gumbel_sample = GumbelSample(temp=0.01)
        sample = get_sample_double(mu, beta, gumbel_sample, 5)
        assert_equal(sample, 0)

    def test_log_likelihood(self):
        mu, beta = 0, 1
        size = 5
        mu_vector = np.ones(size).astype("float32") * mu
        beta_vector = np.ones(size).astype("float32") * beta
        sample = np.zeros(size).astype("float32")
        gumbel_sample = GumbelSample(seed=self.seed, temp=0.1)

        # Directly create theano betaiables instead of using InputLayer.input_beta
        t_mu = T.matrix("mu")
        t_beta = T.matrix("beta")
        t_sample = T.matrix("sample")

        t_log_likelihood = gumbel_sample.log_likelihood(t_sample, t_mu, t_beta)
        f_log_likelihood = theano.function(inputs=[t_sample, t_mu, t_beta], outputs=t_log_likelihood)
        log_likelihood = f_log_likelihood([sample], [mu_vector], [beta_vector])

        scipy_likelihood = scipy.stats.gumbel_r(mu, beta).pdf(sample)
        scipy_log_likelihood = np.log(scipy_likelihood).sum()

        assert_array_almost_equal(log_likelihood, scipy_log_likelihood, decimal=6)


class TestBernoulliSample(TestCase):
    def setUp(self):
        self.seed = 1234567890
        self.bernoulli_sample = BernoulliSample(temp=0.01, seed=1)

    def test_mean_zero(self):
        # Tests the corner case of mean == 0 for the bernoulli distribution.
        # All elements of BernoulliSample.sample(mean=0) should be zero.
        # ref: https://github.com/numpy/numpy/blob/master/numpy/random/tests/test_random.py
        zeros = np.zeros(1000, dtype='float')
        mean = 0
        samples = get_sample(mean, self.bernoulli_sample, 1000)
        assert_(np.allclose(zeros, samples))

    def test_mean_one(self):
        # Tests the corner case of mean == 1 for the bernoulli distribution.
        # All elements of BernoulliSample.sample(mean=1) should be one.
        # ref: https://github.com/numpy/numpy/blob/master/numpy/random/tests/test_random.py
        ones = np.ones(1000, dtype='float')
        mean = 1
        samples = get_sample(mean, self.bernoulli_sample, 1000)
        assert_(np.allclose(ones, samples))

    def test_consistency(self):
        # Ensure that returned values stay the same when setting a fixed seed.
        mean = 0.5
        bernoulli_sample = BernoulliSample(temp=0.1, seed=self.seed)
        actual = get_sample(mean, bernoulli_sample, 5)
        desired = [
            0.9999971508356551,
            0.9101269246280252,
            0.0248156363467501,
            0.3538078165724645,
            0.1615775890919983
        ]
        assert_array_almost_equal(actual, desired, decimal=6)

    def test_log_likelihood(self):
        mean = 0.5
        size = 5
        mean_vector = np.ones(size).astype("float32") * mean
        sample = np.zeros(size).astype("float32")
        bernoulli_sample = BernoulliSample(seed=self.seed, temp=0.01)

        # Directly create theano variables instead of using InputLayer.input_var
        t_mean = T.matrix("mean")
        t_sample = T.matrix("sample")

        t_log_likelihood = bernoulli_sample.log_likelihood(t_sample, t_mean)
        f_log_likelihood = theano.function(inputs=[t_sample, t_mean], outputs=t_log_likelihood)
        log_likelihood = f_log_likelihood([sample], [mean_vector])

        scipy_likelihood = scipy.stats.bernoulli(mean).pmf(sample)
        scipy_log_likelihood = np.log(scipy_likelihood).sum()

        assert_array_almost_equal(log_likelihood[0], scipy_log_likelihood, decimal=6)


class TestGaussianSample(TestCase):
    def setUp(self):
        self.seed = 1234567890

    @staticmethod
    def get_sample(mean, var, gaussian, size):
        # get a sample from given gaussian distribution in ndarray
        mean_vector = np.ones(size).astype("float32") * mean
        var_vector = np.ones(size).astype("float32") * var
        t_mean = T.fvector("mean")  # A theano symbolic variable
        t_var = T.fvector("var")
        t_sample = gaussian.sample(t_mean, t_var)
        f = theano.function(inputs=[t_mean, t_var], outputs=t_sample)
        sample = f(mean_vector, var_vector)
        return sample

    def test_gaussian(self):
        # Test that GaussianSample.sample generates the same result as theano and numpy
        # ref: https://github.com/Theano/Theano/blob/master/theano/tensor/tests/test_shared_randomstreams.py

        gaussian_sample = GaussianSample(seed=utt.fetch_seed())
        mean, var = 0, 1
        tars_sample = get_sample_double(mean, var, gaussian_sample, 5)

        random = RandomStreams(utt.fetch_seed())
        fn = function([], random.normal((5,), mean, var))
        theano_sample = fn()

        rng_seed = np.random.RandomState(utt.fetch_seed()).randint(2**30)
        rng = np.random.RandomState(int(rng_seed))  # int() is for 32bit
        numpy_sample = rng.normal(mean, var, (5,))

        # As GaussianSample.sample method performs reparametrization trick,
        # the return value is slightly different from numpy result. (Reason for setting decimal=7, not 15)
        assert_array_almost_equal(tars_sample, theano_sample, decimal=7)
        assert_array_almost_equal(tars_sample, numpy_sample, decimal=7)

    def test_consistency(self):
        # Ensure that returned values stay the same when setting a fixed seed.
        mean, var = 0, 1
        gaussian_sample = GaussianSample(seed=self.seed)
        actual = get_sample_double(mean, var, gaussian_sample, 5)
        desired = [
            -0.1004791483283043,
            1.2329169511795044,
            -0.1330301910638809,
            1.9520784616470337,
            -0.3533721268177032
        ]
        assert_array_almost_equal(actual, desired, decimal=6)

    def test_mean_zero(self):
        mean, var = 0, 0
        gaussian_sample = GaussianSample()
        sample = get_sample_double(mean, var, gaussian_sample, 5)
        assert_equal(sample, 0)

    def test_log_likelihood(self):
        mean, var = 0, 1
        size = 5
        mean_vector = np.ones(size).astype("float32") * mean
        var_vector = np.ones(size).astype("float32") * var
        sample = np.zeros(size).astype("float32")
        gaussian_sample = GaussianSample(seed=self.seed)

        # Directly create theano variables instead of using InputLayer.input_var
        t_mean = T.matrix("mean")
        t_var = T.matrix("var")
        t_sample = T.matrix("sample")

        t_log_likelihood = gaussian_sample.log_likelihood(t_sample, t_mean, t_var)
        f_log_likelihood = theano.function(inputs=[t_sample, t_mean, t_var], outputs=t_log_likelihood)
        log_likelihood = f_log_likelihood([sample], [mean_vector], [var_vector])

        scipy_likelihood = scipy.stats.norm(mean, var).pdf(sample)
        scipy_log_likelihood = np.log(scipy_likelihood).sum()

        assert_array_almost_equal(log_likelihood, scipy_log_likelihood, decimal=6)


class TestConcreteSample(TestCase):
    def setUp(self):
        self.seed = 1234567890

    def test_consistency(self):
        # Ensure that returned values stay the same when setting a fixed seed.
        mean = 0
        concrete_sample = ConcreteSample(seed=self.seed)
        actual = get_sample(mean, concrete_sample, 5)
        desired = [
            0.9994389867572965,
            0.0000004618787093,
            0.0005598514063278,
            0.0000006999567125,
            0.0000000000009540
        ]
        # TODO: Avoid returning a nested array?
        assert_array_almost_equal(actual[0], desired, decimal=6)

    def test_log_likelihood(self):
        concrete_sample = ConcreteSample()
        self.assertRaises(NotImplementedError, concrete_sample.log_likelihood)


class TestCategoricalSample(TestCase):
    def setUp(self):
        self.seed = 1234567890

    def test_consistency(self):
        # Ensure that returned values stay the same when setting a fixed seed.
        mean = 0
        categorical_sample = CategoricalSample(seed=1234567890)
        actual = get_sample(mean, categorical_sample, 5)
        desired = [
            9.9943900108337402e-01,
            4.6187869884306565e-07,
            5.5985356448218226e-04,
            6.9996337970223976e-07,
            9.5402939020994282e-13
        ]
        assert_array_almost_equal(actual[0], desired, decimal=15)

    def test_log_likelihood(self):
        mean_vector = [0.25]*4
        sample = [0, 1, 0, 0]
        categorical_sample = CategoricalSample(seed=self.seed)

        # Directly create theano variables instead of using InputLayer.input_var
        t_mean = T.matrix("mean")
        t_sample = T.matrix("sample")

        t_log_likelihood = categorical_sample.log_likelihood(t_sample, t_mean)
        f_log_likelihood = theano.function(inputs=[t_sample, t_mean], outputs=t_log_likelihood)
        log_likelihood = f_log_likelihood([sample], [mean_vector])

        scipy_likelihood = scipy.stats.multinomial.pmf(sample, n=1, p=mean_vector)
        scipy_log_likelihood = np.log(scipy_likelihood).sum()

        assert_array_almost_equal(log_likelihood[0], scipy_log_likelihood, decimal=6)


class TestLaplaceSample(TestCase):
    def setUp(self):
        self.seed = 1234567890

    def test_laplace(self):
        # Test LaplaceSample.sample generates the same result as numpy
        # ref: https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.laplace.html

        laplace_sample = LaplaceSample(seed=utt.fetch_seed())
        mean, b = 0, 1
        tars_sample = get_sample_double(mean, b, laplace_sample, 5)

        rng_seed = np.random.RandomState(utt.fetch_seed()).randint(2**30)
        rng = np.random.RandomState(int(rng_seed))  # int() is for 32bit
        numpy_sample = rng.laplace(mean, b, (5,))

        # As laplace_sample.sample method performs reparametrization trick,
        # the return value is slightly different from numpy result. (Reason for setting decimal=7, not 15)
        assert_array_almost_equal(tars_sample, numpy_sample, decimal=7)

    def test_consistency(self):
        # Ensure that returned values stay the same when setting a fixed seed.
        mean = 0
        b = 1
        laplace_sample = LaplaceSample(seed=self.seed)
        actual = get_sample_double(mean, b, laplace_sample, 5)
        desired = [
            1.1390253305435181,
            -0.0570014975965023,
            0.4830891788005829,
            -0.0264561157673597,
            -2.0842888355255127
        ]
        assert_array_almost_equal(actual, desired, decimal=6)

    def test_mean_zero(self):
        mean, b = 0, 0
        laplace_sample = LaplaceSample()
        sample = get_sample_double(mean, b, laplace_sample, 5)
        assert_equal(sample, 0)

    def test_log_likelihood(self):
        mean, b = 0, 1
        size = 5
        mean_vector = np.ones(size).astype("float32") * mean
        b_vector = np.ones(size).astype("float32") * b
        sample = np.zeros(size).astype("float32")
        laplace_sample = LaplaceSample(seed=self.seed)

        # Directly create theano variables instead of using InputLayer.input_var
        t_mean = T.matrix("mean")
        t_b = T.matrix("var")
        t_sample = T.matrix("sample")

        t_log_likelihood = laplace_sample.log_likelihood(t_sample, t_mean, t_b)
        f_log_likelihood = theano.function(inputs=[t_sample, t_mean, t_b], outputs=t_log_likelihood)
        log_likelihood = f_log_likelihood([sample], [mean_vector], [b_vector])

        scipy_likelihood = scipy.stats.laplace(mean, b).pdf(sample)
        scipy_log_likelihood = np.log(scipy_likelihood).sum()

        assert_array_almost_equal(log_likelihood, scipy_log_likelihood, decimal=6)


class TestKumaraswamySample(TestCase):
    def setUp(self):
        self.seed = 1234567890

    def test_consistency(self):
        # Ensure that returned values stay the same when setting a fixed seed.
        a = 0.5
        b = 0.5
        kumaraswamy_sample = KumaraswamySample(seed=self.seed)
        actual = get_sample_double(a, b, kumaraswamy_sample, 5)
        desired = [
            0.0867361798882484,
            0.6036298274993896,
            0.2722139656543732,
            0.5819923281669617,
            0.9922778010368347
        ]
        assert_array_almost_equal(actual, desired, decimal=6)


class TestBetaSample(TestCase):
    def setUp(self):
        self.seed = 1234567890

    def test_consistency(self):
        # Ensure that returned values stay the same when setting a fixed seed.
        a = 0.5
        b = 0.5
        beta_sample = BetaSample(seed=self.seed)
        actual = get_sample_double(a, b, beta_sample, 5)
        desired = [
            0.1251193732023239,
            0.9949938058853149,
            0.8267091512680054,
            0.8865647912025452,
            0.0070835566148162
        ]
        assert_array_almost_equal(actual, desired, decimal=6)

    def test_log_likelihood(self):
        a, b = 0.5, 0.5
        size = 1
        mean_vector = np.ones(size).astype("float32") * a
        var_vector = np.ones(size).astype("float32") * b
        sample = np.ones(size).astype("float32") * 0.5
        beta_sample = BetaSample(seed=self.seed)

        # Directly create theano variables instead of using InputLayer.input_var
        t_a = T.matrix("a")
        t_b = T.matrix("b")
        t_sample = T.matrix("sample")

        t_log_likelihood = beta_sample.log_likelihood(t_sample, t_a, t_b)
        f_log_likelihood = theano.function(inputs=[t_sample, t_a, t_b], outputs=t_log_likelihood)
        log_likelihood = f_log_likelihood([sample], [mean_vector], [var_vector])

        scipy_likelihood = scipy.stats.beta(a, b).pdf(sample)
        scipy_log_likelihood = np.log(scipy_likelihood).sum()

        assert_array_almost_equal(log_likelihood, scipy_log_likelihood, decimal=6)


class TestGammaSample(TestCase):
    def setUp(self):
        self.seed = 1234567890

    def test_consistency(self):
        # Ensure that returned values stay the same when setting a fixed seed.
        a = 0.5
        b = 0.5
        gamma_sample = GammaSample(seed=self.seed)
        actual = get_sample_double(a, b, gamma_sample, 5)
        desired = [
            0.0402308329939842,
            0.9522227048873901,
            1.1623692512512207,
            1.7846230268478394,
            0.0116019621491432
        ]
        assert_array_almost_equal(actual, desired, decimal=6)

    def test_log_likelihood(self):
        a, b = 0.5, 0.5
        size = 1
        a_vector = np.ones(size).astype("float32") * a
        b_vector = np.ones(size).astype("float32") * b
        sample = np.ones(size).astype("float32")
        gamma_sample = GammaSample(seed=self.seed)

        # Directly create theano variables instead of using InputLayer.input_var
        t_a = T.matrix("a")
        t_b = T.matrix("b")
        t_sample = T.matrix("sample")

        t_log_likelihood = gamma_sample.log_likelihood(t_sample, t_a, t_b)
        f_log_likelihood = theano.function(inputs=[t_sample, t_a, t_b], outputs=t_log_likelihood)
        log_likelihood = f_log_likelihood([sample], [a_vector], [b_vector])

        scipy_likelihood = scipy.stats.gamma(a).pdf(sample)
        scipy_log_likelihood = (np.log(scipy_likelihood) + a * np.log(b) + sample - b*sample).sum()

        assert_array_almost_equal(log_likelihood, scipy_log_likelihood, decimal=6)


class TestDirichletSample(TestCase):
    def setUp(self):
        self.seed = 1234567890

    def test_consistency(self):
        # Ensure that returned values stay the same when setting a fixed seed.
        alpha = 0.5
        dirichlet_sample = DirichletSample(k=2, seed=self.seed)
        actual = get_sample(alpha, dirichlet_sample, 5)
        desired = [
            0.1251193732023239,
            0.8748806118965149,
            0.0000000000000000,
            0.0000000000000000,
            0.0000000000000000
        ]
        assert_array_almost_equal(actual, desired, decimal=6)

    def test_log_likelihood(self):
        # size and k need to be the same
        a = 1.2
        size = 2
        a_vector = np.ones(size).astype("float32") * a
        sample = np.ones(size).astype("float32") * 0.5
        dirichlet_sample = DirichletSample(k=2, seed=self.seed)

        # Directly create theano variables instead of using InputLayer.input_var
        t_a = T.fmatrix("a")
        t_sample = T.fmatrix("sample")
        t_log_likelihood = dirichlet_sample.log_likelihood(t_sample, t_a)

        f_log_likelihood = theano.function(inputs=[t_sample, t_a], outputs=t_log_likelihood)
        log_likelihood = f_log_likelihood([sample], [a_vector])

        scipy_likelihood = scipy.stats.dirichlet(a_vector).pdf(sample)
        scipy_log_likelihood = np.log(scipy_likelihood).sum()

        assert_array_almost_equal(log_likelihood, scipy_log_likelihood, decimal=7)
