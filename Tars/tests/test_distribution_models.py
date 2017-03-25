from unittest import TestCase
import mock

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
    BetaSample, GammaSample, DirichletSample, GaussianConstantVarSample
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
from Tars.tests.test_distribution_samples import (
    get_sample, get_sample_double,
    TestBernoulliSample,
    TestGaussianSample,
    TestGumbelSample,
    TestConcreteSample,
    TestCategoricalSample,
    TestLaplaceSample,
    TestKumaraswamySample,
    TestBetaSample,
    TestGammaSample,
    TestDirichletSample
)
from lasagne.layers import InputLayer,DenseLayer
from Tars.distributions import Gaussian
from lasagne.nonlinearities import rectify,linear,softplus


class TestDistribution(TestCase):
    @staticmethod
    def get_samples(mean, size):
        return np.ones(size).astype("float32") * mean

    @staticmethod
    def get_sample_given_x(model, mean_sample):
        t_mean = model.inputs
        t_sample = model.sample_given_x(t_mean)[-1]
        f = theano.function(inputs=t_mean, outputs=t_sample)
        return f([mean_sample])[0]


class TestDistributionDouble(TestCase):
    @staticmethod
    def get_samples(mean, var, size):
        return np.ones(size).astype("float32") * mean, np.ones(size).astype("float32") * var

    @staticmethod
    def get_sample_given_x(model, mean_sample, var_sample, t_mean=None, t_var=None):
        if (t_mean is None) or (t_var is None):
            t_mean, t_var = model.inputs
        sample = model.sample_given_x([t_mean, t_var])[-1]
        f = theano.function(inputs = [t_mean, t_var], outputs=sample)
        return f([mean_sample], [var_sample])[0]


class TestBernoulli(TestCase):
    def setUp(self):
        self.seed = 1234567890
        self.mean = 0
        self.size = 5
        self.mean_sample = TestDistribution.get_samples(self.mean, self.size)
        self.mean_layer = InputLayer((1, None))
        self.model = Bernoulli(self.mean_layer, given=[self.mean_layer], seed=self.seed)
        self.model.set_seed(self.seed)

    @staticmethod
    def get_model(seed=1):
        mean_layer = InputLayer((1, None))
        return Bernoulli(mean_layer, given=[mean_layer], seed=seed)

    def test_sample_given_x_consistency(self):
        # Ensure that returned values stay the same with a fixed seed.
        actual = TestDistribution.get_sample_given_x(
            self.model,
            self.mean_sample,
        )
        desired = [
            2.0341449997965645e-64,
            5.8691278331635298e-69,
            1.4748259625274061e-71,
            3.1732705486287762e-70,
            1.1169141916126556e-70
        ]
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_sample_given_x(self):
        # The output of Bernoulli.sample_given_x should be the same as BernoulliSample.sample
        # when the input is the same, as sample_given_x calls sample inside.
        sample_given_x = TestDistribution.get_sample_given_x(
            self.model,
            self.mean_sample,
        )

        # Pass the same theano variable to BernoulliSample.sample
        bernoulli_sample = BernoulliSample(seed=self.seed)
        sample = get_sample(self.mean, bernoulli_sample, self.size, self.mean_layer.input_var)

        assert_array_almost_equal(sample_given_x, sample, decimal=15)

    def test_np_sample_given_x(self):
        # The output should be the same as sample_given_x
        pass


class TestCategorical(TestCase):
    def setUp(self):
        self.seed = 1234567890
        self.mean = 0
        self.size = 5
        self.mean_sample = TestDistribution.get_samples(self.mean, self.size)
        self.mean_layer = InputLayer((1, self.size))
        self.model = Categorical(self.mean_layer, given=[self.mean_layer], seed=self.seed)
        self.model.set_seed(self.seed)

    @staticmethod
    def get_model(input_size, seed=1):
        mean_layer = InputLayer(input_size)
        return Categorical(mean_layer, given=[mean_layer], seed=seed)

    def test_sample_given_x_consistency(self):
        # Ensure that returned values stay the same with a fixed seed.
        actual = TestDistribution.get_sample_given_x(
            self.model,
            self.mean_sample,
        )
        desired = [
            9.9943898740228387e-01,
            4.6187779994473271e-07,
            5.5985076535571552e-04,
            6.9995360635302726e-07,
            9.5402927277766691e-13
        ]
        assert_array_almost_equal(actual, desired, decimal=15)


    def test_sample_given_x(self):
        # The output of Categorical.sample_given_x should be the same as CategoricalSample.sample
        # when the input is the same, as sample_given_x calls sample inside.
        sample_given_x = TestDistribution.get_sample_given_x(
            self.model,
            self.mean_sample,
        )

        # Pass the same theano variable to CategoricalSample.sample
        categorical_sample = CategoricalSample(seed=self.seed)
        sample = get_sample(self.mean, categorical_sample, self.size, self.mean_layer.input_var)

        assert_array_almost_equal(sample_given_x, sample, decimal=15)


class TestGaussian(TestCase):
    def setUp(self):
        self.seed = 1234567890
        self.mean, self.var = 0, 1
        self.size = 5
        self.mean_sample, self.var_sample = TestDistributionDouble.get_samples(self.mean, self.var, self.size)
        self.mean_layer, self.var_layer = InputLayer((1, None)), InputLayer((1, None))
        self.model = Gaussian(self.mean_layer, self.var_layer, given=[self.mean_layer, self.var_layer], seed=self.seed)
        self.model.set_seed(self.seed)

    @staticmethod
    def get_model(input_size, seed=1):
        mean_layer, var_layer = InputLayer(input_size), InputLayer(input_size)
        return Gaussian(mean_layer, var_layer, given=[mean_layer, var_layer], seed=seed)

    def test_sample_given_x_consistency(self):
        # Ensure that returned values stay the same with a fixed seed.
        actual = TestDistributionDouble.get_sample_given_x(
            self.model,
            self.mean_sample,
            self.var_sample
        )
        desired = [
            -1.0047914519054273e-01,
            1.2329169431414060e+00,
            -1.3303019850182538e-01,
            1.9520784223496075e+00,
            -3.5337213557864172e-01
        ]
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_sample_given_x(self):
        # The output of Gaussian.sample_given_x should be the same as GaussianSample.sample
        # when the input is the same, as sample_given_x calls sample inside.
        sample_given_x = TestDistributionDouble.get_sample_given_x(
            self.model,
            self.mean_sample,
            self.var_sample
        )

        # Pass the same theano variable to BernoulliSample.sample
        gaussian_sample = GaussianSample(seed=self.seed)
        sample = get_sample_double(self.mean, self.var, gaussian_sample, self.size, self.mean_layer.input_var, self.var_layer.input_var)

        assert_array_almost_equal(sample_given_x, sample, decimal=15)


    def test_np_sample_given_x(self):
        # The output should be the same as sample_given_x
        pass


class TestGaussianConstantVar(TestCase):
    def setUp(self):
        self.seed = 1234567890
        self.mean = 0
        self.size = 5
        self.mean_sample = TestDistribution.get_samples(self.mean, self.size)
        self.mean_layer = InputLayer((1, None))
        self.model = GaussianConstantVar(self.mean_layer, given=[self.mean_layer], seed=self.seed)
        self.model.set_seed(self.seed)

    @staticmethod
    def get_model(input_size, seed=1):
        mean_layer = InputLayer(input_size)
        return GaussianConstantVar(mean_layer, given=[mean_layer], seed=seed)

    def test_sample_given_x_consistency(self):
        # Ensure that returned values stay the same with a fixed seed.
        actual = TestDistribution.get_sample_given_x(
            self.model,
            self.mean_sample,
        )
        desired = [
            -1.0047914519054273e-01,
            1.2329169431414060e+00,
            -1.3303019850182538e-01,
            1.9520784223496075e+00,
            -3.5337213557864172e-01
        ]
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_sample_given_x(self):
        # The output of GaussianConstantVar.sample_given_x should be the same as GaussianConstantVarSample.sample
        # when the input is the same, as sample_given_x calls sample inside.
        sample_given_x = TestDistribution.get_sample_given_x(
            self.model,
            self.mean_sample,
        )

        # Pass the same theano variable to GaussianConstantVarSample.sample
        g_constant_var_sample = GaussianConstantVarSample(seed=self.seed)
        sample = get_sample(self.mean, g_constant_var_sample, self.size, self.mean_layer.input_var)

        assert_array_almost_equal(sample_given_x, sample, decimal=15)


class TestLaplace(TestCase):
    def setUp(self):
        self.seed = 1234567890
        self.mean, self.var = 0, 1
        self.size = 5
        self.mean_sample, self.var_sample = TestDistributionDouble.get_samples(self.mean, self.var, self.size)
        self.input_size = (1, self.size)
        self.mean_layer, self.var_layer = InputLayer(self.input_size), InputLayer(self.input_size)
        self.model = Laplace(self.mean_layer, self.var_layer, given=[self.mean_layer, self.var_layer], seed=self.seed)
        self.model.set_seed(self.seed)

    @staticmethod
    def get_model(input_size, seed=1):
        mean_layer, var_layer = InputLayer(input_size), InputLayer(input_size)
        return Laplace(mean_layer, var_layer, given=[mean_layer, var_layer], seed=seed)

    def test_sample_given_x_consistency(self):
        # Ensure that returned values stay the same with a fixed seed.
        actual = TestDistributionDouble.get_sample_given_x(
            self.model,
            self.mean_sample,
            self.var_sample
        )
        desired = [
            1.1390253664399406e+00,
            -5.7001508991036981e-02,
            4.8308916318957201e-01,
            -2.6456103460765695e-02,
            -2.0842891122384506e+00
        ]
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_sample_given_x(self):
        # The output of Laplace.sample_given_x should be the same as LaplaceSample.sample
        # when the input is the same, as sample_given_x calls sample inside.
        sample_given_x = TestDistributionDouble.get_sample_given_x(
            self.model,
            self.mean_sample,
            self.var_sample
        )

        # Pass the same theano variable to LaplaceSample.sample
        laplace_sample = LaplaceSample(seed=self.seed)
        sample = get_sample_double(self.mean, self.var, laplace_sample, self.size, self.mean_layer.input_var, self.var_layer.input_var)

        assert_array_almost_equal(sample_given_x, sample, decimal=15)


class TestGamma(TestCase):
    def setUp(self):
        self.seed = 1234567890
        self.alpha, self.beta = 0.5, 0.5
        self.size = 5
        self.alpha_sample, self.beta_sample = TestDistributionDouble.get_samples(self.alpha, self.beta, self.size)
        self.alpha_layer, self.beta_layer = InputLayer((1, None)), InputLayer((1, None))
        self.model = Gamma(self.alpha_layer, self.beta_layer, given=[self.alpha_layer, self.beta_layer], seed=self.seed)
        self.model.set_seed(self.seed)

    @staticmethod
    def get_model(input_size, seed=1):
        alpha_layer, beta_layer = InputLayer(input_size), InputLayer(input_size)
        return Gamma(alpha_layer, beta_layer, given=[alpha_layer, beta_layer], seed=seed)

    def test_sample_given_x_consistency(self):
        # Ensure that returned values stay the same with a fixed seed.
        actual = TestDistributionDouble.get_sample_given_x(
            self.model,
            self.alpha_sample,
            self.beta_sample
        )
        desired = [
            4.0230831641623901e-02,
            9.5222264503004839e-01,
            1.1623693168813076e+00,
            1.7846231234738394e+00,
            1.1601962737344840e-02
        ]
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_sample_given_x(self):
        # The output of Gamma.sample_given_x should be the same as GammaSample.sample
        # when the input is the same, as sample_given_x calls sample inside.
        sample_given_x = TestDistributionDouble.get_sample_given_x(
            self.model,
            self.alpha_sample,
            self.beta_sample
        )

        # Pass the same theano variable to GammaSample.sample
        gamma_sample = GammaSample(seed=self.seed)
        sample = get_sample_double(self.alpha, self.beta, gamma_sample, self.size, self.alpha_layer.input_var, self.beta_layer.input_var)

        assert_array_almost_equal(sample_given_x, sample, decimal=15)


class TestBeta(TestCase):
    def setUp(self):
        self.seed = 1234567890
        self.alpha, self.beta = 0.5, 0.5
        self.size = 5
        self.alpha_sample, self.beta_sample = TestDistributionDouble.get_samples(self.alpha, self.beta, self.size)
        self.alpha_layer, self.beta_layer = InputLayer((1, None)), InputLayer((1, None))
        self.model = Beta(self.alpha_layer, self.beta_layer, given=[self.alpha_layer, self.beta_layer], seed=self.seed)
        self.model.set_seed(self.seed)

    @staticmethod
    def get_model(input_size, seed=1):
        alpha_layer, beta_layer = InputLayer(input_size),InputLayer(input_size)
        return Beta(alpha_layer, beta_layer, given=[alpha_layer, beta_layer], seed=seed)

    def test_sample_given_x_consistency(self):
        # Ensure that returned values stay the same with a fixed seed.
        actual = TestDistributionDouble.get_sample_given_x(
            self.model,
            self.alpha_sample,
            self.beta_sample
        )
        desired = [
            1.2511939123119001e-01,
            9.9499377523712407e-01,
            8.2670919515752916e-01,
            8.8656480966410656e-01,
            7.0835564247564431e-03
        ]
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_sample_given_x(self):
        # The output of Beta.sample_given_x should be the same as BetaSample.sample
        # when the input is the same, as sample_given_x calls sample inside.
        sample_given_x = TestDistributionDouble.get_sample_given_x(
            self.model,
            self.alpha_sample,
            self.beta_sample
        )

        # Pass the same theano variable to GammaSample.sample
        beta_dist_sample = BetaSample(seed=self.seed)
        sample = get_sample_double(self.alpha, self.beta, beta_dist_sample, self.size, self.alpha_layer.input_var, self.beta_layer.input_var)

        assert_array_almost_equal(sample_given_x, sample, decimal=15)


class TestDirichlet(TestCase):
    def setUp(self):
        self.seed = 1234567890
        self.alpha = 0.5
        self.size = 3
        self.k = 3
        self.alpha_sample = TestDistribution.get_samples(self.alpha, self.size) # k and the length of samples must be the same
        self.alpha_layer = InputLayer((1, self.size))
        self.model = Dirichlet(self.alpha_layer, given=[self.alpha_layer], k=self.k, iter_sampling=1, seed=self.seed)
        self.model.set_seed(self.seed)

    @staticmethod
    def get_model(input_size, k=2, seed=1):
        alpha_layer = InputLayer(input_size)
        return Dirichlet(alpha_layer, given=[alpha_layer], k=k, seed=seed)

    def test_sample_given_x_consistency(self):
        # Ensure that returned values stay the same with a fixed seed.
        actual = TestDistribution.get_sample_given_x(
            self.model,
            self.alpha_sample,
        )
        desired = [
            1.1092685383468495e-01,
            0.0000000000000000e+00,
            8.8907314616531508e-01
        ]
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_sample_given_x(self):
        # The output of Dirichletsample_given_x should be the same as DirichletSample.sample
        # when the input is the same, as sample_given_x calls sample inside.
        sample_given_x = TestDistribution.get_sample_given_x(
            self.model,
            self.alpha_sample,
        )

        # Pass the same theano variable to DirichletSample.sample
        dirichlet_sample = DirichletSample(self.k, seed=self.seed)
        sample = get_sample(self.alpha, dirichlet_sample, self.size, self.alpha_layer.input_var)

        assert_array_almost_equal(sample_given_x, sample, decimal=15)


def display_samples(samples, format='e'):
    for i in samples:
        print ('{0:.16%s}'%format).format(i)
    print '='*20

