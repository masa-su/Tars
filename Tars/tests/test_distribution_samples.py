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
    @staticmethod
    def get_sample(mu, beta, gumbel, size):
        # get a sample from given gumbel distribution in ndarray
        mu_vector = np.ones(size).astype("float32") * mu
        beta_vector = np.ones(size).astype("float32") * beta
        t_mu = T.fvector("mu")
        t_beta = T.fvector("beta")
        t_sample = gumbel.sample(t_mu, t_beta)
        f = theano.function(inputs=[t_mu, t_beta], outputs=t_sample)
        sample = f(mu_vector, beta_vector)
        return sample

    def test_consistency(self):
        # Ensure that returned values stay the same when setting a fixed seed.
        mu, beta = 0, 1
        gumbel_sample = GumbelSample(temp=0.01, seed=1234567890)
        actual = TestGumbelSample.get_sample(mu, beta, gumbel_sample, 5)
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
        sample = TestGumbelSample.get_sample(mu, beta, gumbel_sample, 5)
        assert_equal(sample, 0)


class TestBernoulliSample(TestCase):
    def setUp(self):
        self.bernoulli_sample = BernoulliSample(temp=0.01, seed=1)

    @staticmethod
    def get_sample(mean, bernoulli, size, t_mean=None):
        if t_mean is None:
            t_mean = T.fvector("mean")
        # get a sample from given bernoulli distribution in ndarray
        mean_vector = np.ones(size).astype("float32") * mean
        t_sample = bernoulli.sample(t_mean)
        f = theano.function(inputs=[t_mean], outputs=t_sample)

        if t_mean.ndim == 1:
            return f(mean_vector)
        elif t_mean.ndim == 2:
            return f([mean_vector])[0]

    def test_mean_zero(self):
        # Tests the corner case of mean == 0 for the bernoulli distribution.
        # All elements of BernoulliSample.sample(mean=0) should be zero.
        # ref: https://github.com/numpy/numpy/blob/master/numpy/random/tests/test_random.py
        zeros = np.zeros(1000, dtype='float')
        mean = 0
        samples = TestBernoulliSample.get_sample(mean, self.bernoulli_sample, 1000)
        assert_(np.allclose(zeros, samples))

    def test_mean_one(self):
        # Tests the corner case of mean == 1 for the bernoulli distribution.
        # All elements of BernoulliSample.sample(mean=1) should be one.
        # ref: https://github.com/numpy/numpy/blob/master/numpy/random/tests/test_random.py
        ones = np.ones(1000, dtype='float')
        mean = 1
        samples = TestBernoulliSample.get_sample(mean, self.bernoulli_sample, 1000)
        assert_(np.allclose(ones, samples))

    def test_consistency(self):
        # Ensure that returned values stay the same when setting a fixed seed.
        mean = 0.5
        bernoulli_sample = BernoulliSample(temp=0.1, seed=1234567890)
        actual = TestBernoulliSample.get_sample(mean, bernoulli_sample, 5)
        desired = [
            0.9999971508356551,
            0.9101269246280252,
            0.0248156363467501,
            0.3538078165724645,
            0.1615775890919983
        ]
        display_samples(actual)
        assert_array_almost_equal(actual, desired, decimal=6)


class TestGaussianSample(TestCase):
    def setUp(self):
        pass

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
        tars_sample = TestGaussianSample.get_sample(mean, var, gaussian_sample, 5)

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
        gaussian_sample = GaussianSample(seed=1234567890)
        actual = TestGaussianSample.get_sample(mean, var, gaussian_sample, 5)
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
        sample = TestGaussianSample.get_sample(mean, var, gaussian_sample, 5)
        assert_equal(sample, 0)

    def test_log_likelihood(self):
        import lasagne
        from lasagne.layers import InputLayer,DenseLayer
        from Tars.distributions.distribution_models import Gaussian

        seed = utt.fetch_seed()
        print seed
        mean, var = 0, 1
        size = 5
        mean_vector = np.ones(size).astype("float32") * mean
        var_vector = np.ones(size).astype("float32") * var

        gaussian_sample = GaussianSample(seed=seed)
        t_mean = T.fvector("mean")  # A theano symbolic variable
        t_var = T.fvector("var")
        t_sample = T.fvector("sample")
        print t_sample.ndim

        #t_sample = gaussian_sample.sample(t_mean, t_var)
        #f = theano.function(inputs=[t_mean, t_var], outputs=t_sample)#
        #sample = f(mean_vector, var_vector)
        #print sample.shape, sample

        mean_layer, var_layer = InputLayer((1, None)), InputLayer((1, None))
        model = Gaussian(mean_layer, var_layer, given=[mean_layer, var_layer], seed=seed)
        t_output = model.fprop(model.inputs)
        print model.get_input_shape()
        print model.get_output_shape()
        #mean_vector = mean_vector.reshape(mean_vector.shape[0],1)
        #var_vector = var_vector.reshape(var_vector.shape[0],1)

        f_output = theano.function(inputs=list(model.inputs), outputs=t_output)
        output = f_output([mean_vector], [var_vector])
        print output
        print output[0].ndim
        print output[1].ndim


        t_ll = gaussian_sample.log_likelihood(t_sample, t_output[0], t_output[1])
        print t_ll

        f_ll = theano.function(inputs=[t_sample, t_output[0], t_output[1]], outputs=t_ll)
        ll = f_ll([0,0,0,0,0], [mean_vector], [var_vector])
        print ll
        for i in ll:
            print '{0:.16f}'.format(i)
        print '='*10

        from scipy.stats import norm
        lh = norm(mean, var).pdf([0,0,0,0,0])
        scipy_ll = np.log(lh).sum()
        print '{0:.16f}'.format(scipy_ll)


class TestConcreteSample(TestCase):

    @staticmethod
    def get_sample(mean, concrete, size):
        # get a sample from given concrete distribution in ndarray
        mean_vector = np.ones(size).astype("float32") * mean
        t_mean = T.fvector("mean")  # A theano symbolic variable
        t_sample = concrete.sample(t_mean)
        f = theano.function(inputs=[t_mean], outputs=t_sample)
        sample = f(mean_vector)
        return sample

    def test_consistency(self):
        # Ensure that returned values stay the same when setting a fixed seed.
        mean = 0
        concrete_sample = ConcreteSample(seed=1234567890)
        actual = TestConcreteSample.get_sample(mean, concrete_sample, 5)
        desired = [
            0.9994389867572965,
            0.0000004618787093,
            0.0005598514063278,
            0.0000006999567125,
            0.0000000000009540
        ]
        # TODO: Avoid returning a nested array?
        assert_array_almost_equal(actual[0], desired, decimal=6)


class TestCategoricalSample(TestCase):

    @staticmethod
    def get_sample(mean, categorical, size, t_mean=None):
        if t_mean is None:
            t_mean = T.fvector("mean")
        # get a sample from given categorical distribution in ndarray
        mean_vector = np.ones(size).astype("float32") * mean
        t_sample = categorical.sample(t_mean)
        f = theano.function(inputs=[t_mean], outputs=t_sample)

        if t_mean.ndim == 1:
            return f(mean_vector)
        elif t_mean.ndim == 2:
            return f([mean_vector])[0]

    def test_consistency(self):
        # Ensure that returned values stay the same when setting a fixed seed.
        mean = 0
        categorical_sample = CategoricalSample(seed=1234567890)
        actual = TestCategoricalSample.get_sample(mean, categorical_sample, 10)
        print actual
        desired = [
            0.9994389867572965,
            0.0000004618787093,
            0.0005598514063278,
            0.0000006999567125,
            0.0000000000009540
        ]
        for i in actual[0]:
            print '{0:.16e}'.format(i)
        # TODO: Avoid returning a nested array?
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_consistency2(self):
        mean = 0


class TestLaplaceSample(TestCase):

    @staticmethod
    def get_sample(mean, b, laplace, size):
        # get a sample from given laplace distribution in ndarray
        mean_vector = np.ones(size).astype("float32") * mean
        b_vector = np.ones(size).astype("float32") * b
        t_mean = T.fvector("mean")  # A theano symbolic variable
        t_b = T.fvector("b")
        t_sample = laplace.sample(t_mean, t_b)
        f = theano.function(inputs=[t_mean, t_b], outputs=t_sample)
        sample = f(mean_vector, b_vector)
        return sample

    def test_laplace(self):
        # Test LaplaceSample.sample generates the same result as numpy
        # ref: https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.laplace.html

        laplace_sample = LaplaceSample(seed=utt.fetch_seed())
        mean, b = 0, 1
        tars_sample = TestLaplaceSample.get_sample(mean, b, laplace_sample, 5)

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
        laplace_sample = LaplaceSample(seed=1234567890)
        actual = TestLaplaceSample.get_sample(mean, b, laplace_sample, 5)
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
        sample = TestLaplaceSample.get_sample(mean, b, laplace_sample, 5)
        assert_equal(sample, 0)


class TestKumaraswamySample(TestCase):

    @staticmethod
    def get_sample(a, b, kumaraswamy, size):
        # get a sample from given kumaraswamy distribution in ndarray
        a_vector = np.ones(size).astype("float32") * a
        b_vector = np.ones(size).astype("float32") * b
        t_a = T.fvector("a")  # A theano symbolic variable
        t_b = T.fvector("b")
        t_sample = kumaraswamy.sample(t_a, t_b)
        f = theano.function(inputs=[t_a, t_b], outputs=t_sample)
        sample = f(a_vector, b_vector)
        return sample

    def test_consistency(self):
        # Ensure that returned values stay the same when setting a fixed seed.
        a = 0.5
        b = 0.5
        kumaraswamy_sample = KumaraswamySample(seed=1234567890)
        actual = TestKumaraswamySample.get_sample(a, b, kumaraswamy_sample, 5)
        desired = [
            0.0867361798882484,
            0.6036298274993896,
            0.2722139656543732,
            0.5819923281669617,
            0.9922778010368347
        ]
        assert_array_almost_equal(actual, desired, decimal=6)


class TestBetaSample(TestCase):

    @staticmethod
    def get_sample(a, b, beta, size):
        # get a sample from given beta distribution in ndarray
        a_vector = np.ones(size).astype("float32") * a
        b_vector = np.ones(size).astype("float32") * b
        t_a = T.fvector("a")  # A theano symbolic variable
        t_b = T.fvector("b")
        t_sample = beta.sample(t_a, t_b)
        f = theano.function(inputs=[t_a, t_b], outputs=t_sample)
        sample = f(a_vector, b_vector)
        return sample

    def test_consistency(self):
        # Ensure that returned values stay the same when setting a fixed seed.
        a = 0.5
        b = 0.5
        beta_sample = BetaSample(seed=1234567890)
        actual = TestBetaSample.get_sample(a, b, beta_sample, 5)
        desired = [
            0.1251193732023239,
            0.9949938058853149,
            0.8267091512680054,
            0.8865647912025452,
            0.0070835566148162
        ]
        assert_array_almost_equal(actual, desired, decimal=6)


class TestGammaSample(TestCase):

    @staticmethod
    def get_sample(a, b, gamma, size):
        # get a sample from given gamma distribution in ndarray
        a_vector = np.ones(size).astype("float32") * a
        b_vector = np.ones(size).astype("float32") * b
        t_a = T.fvector("a")  # A theano symbolic variable
        t_b = T.fvector("b")
        t_sample = gamma.sample(t_a, t_b)
        f = theano.function(inputs=[t_a, t_b], outputs=t_sample)
        sample = f(a_vector, b_vector)
        return sample

    def test_consistency(self):
        # Ensure that returned values stay the same when setting a fixed seed.
        a = 0.5
        b = 0.5
        gamma_sample = GammaSample(seed=1234567890)
        actual = TestGammaSample.get_sample(a, b, gamma_sample, 5)
        desired = [
            0.0402308329939842,
            0.9522227048873901,
            1.1623692512512207,
            1.7846230268478394,
            0.0116019621491432
        ]
        assert_array_almost_equal(actual, desired, decimal=6)


class TestDirichletSample(TestCase):

    @staticmethod
    def get_sample(alpha, dirichlet, size):
        # get a sample from given dirichlet distribution in ndarray
        alpha_vector = np.ones(size).astype("float32") * alpha
        t_alpha = T.fvector("alpha")
        t_sample = dirichlet.sample(t_alpha)
        f = theano.function(inputs=[t_alpha], outputs=t_sample)
        sample = f(alpha_vector)
        return sample

    def test_consistency(self):
        # Ensure that returned values stay the same when setting a fixed seed.
        alpha = 100
        dirichlet_sample = DirichletSample(k=2, seed=1234567890)
        actual = TestDirichletSample.get_sample(alpha, dirichlet_sample, 5)
        desired = [
            0.1251193732023239,
            0.8748806118965149,
            0.0000000000000000,
            0.0000000000000000,
            0.0000000000000000
        ]
        assert_array_almost_equal(actual, desired, decimal=6)


def display_samples(samples, format=''):
    for i in samples:
        #print '{0:.16f}'.format(i)
        print '{0:.16e}'.format(i)