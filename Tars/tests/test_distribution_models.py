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
from Tars.tests.test_distribution_samples import (
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
    def get_samples(mean, size=10):
        return np.ones(size).astype("float32") * mean

    @staticmethod
    def get_sample_given_x(model, mean_sample):
        t_mean = model.inputs
        sample = model.sample_given_x(t_mean)[-1]
        f = theano.function(inputs=t_mean, outputs=sample)
        return f([mean_sample])[0]


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


class TestBernoulli(TestCase):
    def setUp(self):
        self.seed = 1234567890
        self.mean = 0
        self.size = 10
        self.mean_sample = TestDistribution.get_samples(self.mean)
        self.model = TestBernoulli.get_model(seed=self.seed)
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
            2.0341405545125100e-64,
            5.8691140929553508e-69,
            1.4748223782549935e-71,
            3.1732629817326694e-70,
            1.1169115108701376e-70,
            3.5682764041735962e-58,
            1.0438842879901357e-59,
            9.4375179159552933e-73,
            2.0604359778275096e-64,
            1.8783024095913569e-63
        ]
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_sample_given_x(self):
        # The output should be the same as BernoulliSample.sample when given equals to the input.
        sample_given_x = TestDistribution.get_sample_given_x(
            self.model,
            self.mean_sample,
        )
        # Generate samples from BernoulliSample with the same seed as the model
        bernoulli_sample = BernoulliSample(seed=self.seed)
        sample = TestBernoulliSample.get_sample(self.mean, bernoulli_sample, self.size)
        assert_array_almost_equal(sample_given_x, sample, decimal=15)

    def test_np_sample_given_x(self):
        # The output should be the same as sample_given_x
        pass


class TestCategorical(TestCase):
    @staticmethod
    def get_model(seed=1):
        mean_layer = InputLayer((1, 10))
        return Categorical(mean_layer, given=[mean_layer], seed=seed)

    def test_sample_given_x_consistency(self):
        # Ensure that returned values stay the same with a fixed seed.
        seed = 1234567890
        mean = 0
        mean_sample = TestDistribution.get_samples(mean)

        model = TestCategorical.get_model(seed=seed)
        model.set_seed(seed)

        actual = TestDistribution.get_sample_given_x(
            model,
            mean_sample,
        )
        desired = [
            2.9662526142361950e-09,
            1.3708149782097527e-15,
            1.6615907844025203e-12,
            2.0774042261811396e-15,
            2.8314791527853584e-21,
            9.9999999690715935e-01,
            1.2491817031036337e-10,
            3.9659659863133490e-16,
            1.3480502321912626e-15,
            3.1816204529416427e-15
        ]
        assert_array_almost_equal(actual, desired, decimal=15)


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


class TestGaussianConstantVar(TestCase):
    @staticmethod
    def get_model(seed=1):
        mean_layer = InputLayer((1, None))
        return GaussianConstantVar(mean_layer, given=[mean_layer], seed=seed)

    def test_sample_given_x_consistency(self):
        # Ensure that returned values stay the same with a fixed seed.
        seed = 1234567890
        mean = 0
        mean_sample = TestDistribution.get_samples(mean)

        model = TestGaussianConstantVar.get_model(seed=seed)
        model.set_seed(seed)

        actual = TestDistribution.get_sample_given_x(
            model,
            mean_sample,
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


class TestLaplace(TestCase):
    @staticmethod
    def get_model(seed=1):
        mean_layer, var_layer = InputLayer((1, None)),InputLayer((1, None))
        return Laplace(mean_layer, var_layer, given=[mean_layer, var_layer], seed=seed)

    def test_sample_given_x_consistency(self):
        # Ensure that returned values stay the same with a fixed seed.
        seed = 1234567890
        mean, var = 0, 1
        mean_sample, var_sample = TestDistributionDouble.get_samples(mean, var)

        model = TestLaplace.get_model(seed=seed)
        model.set_seed(seed)

        actual = TestDistributionDouble.get_sample_given_x(
            model,
            mean_sample,
            var_sample
        )
        desired = [
            1.1390253664399406e+00,
            -5.7001508991036981e-02,
            4.8308916318957201e-01,
            -2.6456103460765695e-02,
            -2.0842891122384506e+00,
            3.0288897904175491e+00,
            8.5366618427015917e-01,
            -1.5605352232930525e-01,
            -5.8258772622256469e-02,
            3.5867596993898196e-03
        ]
        assert_array_almost_equal(actual, desired, decimal=15)


class TestGamma(TestCase):
    @staticmethod
    def get_model(seed=1):
        alpha_layer, beta_layer = InputLayer((1, None)),InputLayer((1, None))
        return Gamma(alpha_layer, beta_layer, given=[alpha_layer, beta_layer], seed=seed)

    def test_sample_given_x_consistency(self):
        # Ensure that returned values stay the same with a fixed seed.
        seed = 1234567890
        alpha, beta = 0.5, 0.5
        alpha_sample, beta_sample = TestDistributionDouble.get_samples(alpha, beta)

        model = TestGamma.get_model(seed=seed)
        model.set_seed(seed)

        actual = TestDistributionDouble.get_sample_given_x(
            model,
            alpha_sample,
            beta_sample
        )
        desired = [
            4.0230831641623901e-02,
            9.5222264503004839e-01,
            1.1623693168813076e+00,
            1.7846231234738394e+00,
            1.1601962737344840e-02,
            1.2021816815566539e+00,
            1.7568047926338815e+00,
            2.1646092000939429e-01,
            4.2426224339589874e-01,
            2.6116683924547507e-01
        ]
        assert_array_almost_equal(actual, desired, decimal=15)


class TestBeta(TestCase):
    @staticmethod
    def get_model(seed=1):
        alpha_layer, beta_layer = InputLayer((1, None)),InputLayer((1, None))
        return Beta(alpha_layer, beta_layer, given=[alpha_layer, beta_layer], seed=seed)

    def test_sample_given_x_consistency(self):
        # Ensure that returned values stay the same with a fixed seed.
        seed = 1234567890
        alpha, beta = 0.5, 0.5
        alpha_sample, beta_sample = TestDistributionDouble.get_samples(alpha, beta)

        model = TestBeta.get_model(seed=seed)
        model.set_seed(seed)

        actual = TestDistributionDouble.get_sample_given_x(
            model,
            alpha_sample,
            beta_sample
        )
        desired = [
            1.2511939123119001e-01,
            9.9499377523712407e-01,
            8.2670919515752916e-01,
            8.8656480966410656e-01,
            7.0835564247564431e-03,
            9.2395593282207600e-01,
            5.1113276073478231e-01,
            8.0123533933310609e-01,
            3.0581190019222942e-01,
            9.8989224450713886e-01
        ]
        assert_array_almost_equal(actual, desired, decimal=15)


class TestDirichlet(TestCase):
    @staticmethod
    def get_model(k=2, seed=1):
        alpha_layer = InputLayer((1, None))
        return Dirichlet(alpha_layer, given=[alpha_layer], k=k, seed=seed)

    def test_sample_given_x_consistency(self):
        # Ensure that returned values stay the same with a fixed seed.
        seed = 1234567890
        alpha = 0.5
        k = 2
        alpha_sample = TestDistribution.get_samples(alpha)

        model = TestDirichlet.get_model(k=k, seed=seed)
        model.set_seed(seed)

        actual = TestDistribution.get_sample_given_x(
            model,
            alpha_sample,
        )
        desired = [
            1.2511939123119001e-01,
            8.7488060876881002e-01,
            9.9499377523712407e-01,
            5.0062247628759257e-03,
            8.2670919515752916e-01,
            1.7329080484247078e-01,
            8.8656480966410656e-01,
            1.1343519033589346e-01,
            7.0835564247564431e-03,
            9.9291644357524356e-01
        ]
        assert_array_almost_equal(actual, desired, decimal=15)
