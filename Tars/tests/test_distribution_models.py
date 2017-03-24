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
        t_sample = model.sample_given_x(t_mean)[-1]
        f = theano.function(inputs=t_mean, outputs=t_sample)
        return f([mean_sample])[0]
        #return f(mean_sample)


class TestDistributionDouble(TestCase):
    @staticmethod
    def get_samples(mean, var, size=10):
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
        self.size = 10
        self.mean_sample = TestDistribution.get_samples(self.mean)
        self.model = TestBernoulli.get_model(seed=self.seed)
        self.model.set_seed(self.seed)

    @staticmethod
    def get_model(seed=1):
        mean_layer = InputLayer((1, None))
        #mean_layer = InputLayer((None,))
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
        # Generate samples from BernoulliSample with the same seed as the model
        #mean_layer = InputLayer((None,)) # ndim==1
        mean_layer = InputLayer((1, None)) # ndim==2
        self.model = Bernoulli(mean_layer, given=[mean_layer], seed=self.seed)
        self.model.set_seed(self.seed)
        sample_given_x = TestDistribution.get_sample_given_x(
            self.model,
            self.mean_sample,
        )
        bernoulli_sample = BernoulliSample(seed=self.seed)
        #sample = TestBernoulliSample.get_sample(self.mean, bernoulli_sample, self.size)
        display_samples(sample_given_x)

        # Pass the same theano variable to BernoulliSample.sample
        t_mean = mean_layer.input_var
        mean_vector = np.ones(self.size).astype("float32") * self.mean
        t_sample = bernoulli_sample.sample(t_mean)
        f = theano.function(inputs=[t_mean], outputs=t_sample)
        #sample = f(mean_vector)
        sample = f([mean_vector])[0]
        display_samples(sample)

        assert_array_almost_equal(sample_given_x, sample, decimal=15)

    def test_np_sample_given_x(self):
        # The output should be the same as sample_given_x
        pass


class TestCategorical(TestCase):
    def setUp(self):
        self.seed = 1234567890
        self.mean = 0
        self.size = 10
        self.mean_sample = TestDistribution.get_samples(self.mean)
        self.model = TestCategorical.get_model((1, self.size), seed=self.seed)
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


    def test_sample_given_x(self):
        # The output should be the same as CategoricalSample.sample,
        # when the shape of inputs is the same.
        mean_layer = InputLayer((1, self.size)) # ndim==2
        self.model = Categorical(mean_layer, given=[mean_layer], seed=self.seed)
        self.model.set_seed(self.seed)
        sample_given_x = TestDistribution.get_sample_given_x(
            self.model,
            self.mean_sample,
        )
        categorical_sample = CategoricalSample(seed=self.seed)
        display_samples(sample_given_x)

        # Pass the same theano variable to BernoulliSample.sample
        t_mean = mean_layer.input_var
        mean_vector = np.ones(self.size).astype("float32") * self.mean
        t_sample = categorical_sample.sample(t_mean)
        f = theano.function(inputs=[t_mean], outputs=t_sample)
        #sample = f(mean_vector)
        sample = f([mean_vector])[0]
        display_samples(sample)

        assert_array_almost_equal(sample_given_x, sample, decimal=15)


class TestGaussian(TestCase):
    def setUp(self):
        self.seed = 1234567890
        self.mean, self.var = 0, 1
        self.size = 10
        self.mean_sample, self.var_sample = TestDistributionDouble.get_samples(self.mean, self.var)
        self.model = TestGaussian.get_model((1, None), seed=self.seed)
        self.model.set_seed(self.seed)

    @staticmethod
    def get_model(input_size, seed=1):
        mean_layer, var_layer = InputLayer(input_size), InputLayer(input_size)
        return Gaussian(mean_layer, var_layer, given=[mean_layer, var_layer], seed=seed)

    def test_sample_given_x_consistency(self):
        # Ensure that returned values stay the same with a fixed seed.
        #mean_sample, var_sample = TestDistributionDouble.get_samples(self.mean, self.var)

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
        mean_layer = InputLayer((1, None)) # ndim==2
        var_layer = InputLayer((1, None))
        self.model = Gaussian(mean_layer, var_layer, given=[mean_layer, var_layer], seed=self.seed)
        self.model.set_seed(self.seed)
        sample_given_x = TestDistributionDouble.get_sample_given_x(
            self.model,
            self.mean_sample,
            self.var_sample
        )
        display_samples(sample_given_x)

        gaussian_sample = GaussianSample(seed=self.seed)
        t_mean = mean_layer.input_var                       # Pass the same theano variable to BernoulliSample.sample
        t_var = var_layer.input_var
        mean_vector = np.ones(self.size).astype("float32") * self.mean
        var_vector = np.ones(self.size).astype("float32") * self.var
        t_sample = gaussian_sample.sample(t_mean, t_var)
        f = theano.function(inputs=[t_mean, t_var], outputs=t_sample)
        #sample = f(mean_vector)
        sample = f([mean_vector], [var_vector])[0]
        display_samples(sample)

        assert_array_almost_equal(sample_given_x, sample, decimal=15)


    def test_np_sample_given_x(self):
        # The output should be the same as sample_given_x
        pass


class TestGaussianConstantVar(TestCase):
    def setUp(self):
        self.seed = 1234567890
        self.mean = 0
        self.size = 10
        self.mean_sample = TestDistribution.get_samples(self.mean)
        self.model = TestGaussianConstantVar.get_model((1, None), seed=self.seed)
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
            -3.5337213557864172e-01,
            1.4042016515710956e+00,
            2.1391636298546729e-01,
            -3.3813674071571591e+00,
            -9.8588173289413522e-02,
            1.9687688060042323e+00
        ]
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_sample_given_x(self):
        # The output should be the same as CategoricalSample.sample,
        # when the shape of inputs is the same.
        mean_layer = InputLayer((1, self.size)) # ndim==2
        self.model = GaussianConstantVar(mean_layer, given=[mean_layer], seed=self.seed)
        self.model.set_seed(self.seed)
        sample_given_x = TestDistribution.get_sample_given_x(
            self.model,
            self.mean_sample,
        )
        g_constant_var_sample = GaussianConstantVarSample(seed=self.seed)
        display_samples(sample_given_x)

        # Pass the same theano variable to BernoulliSample.sample
        t_mean = mean_layer.input_var
        mean_vector = np.ones(self.size).astype("float32") * self.mean
        t_sample = g_constant_var_sample.sample(t_mean)
        f = theano.function(inputs=[t_mean], outputs=t_sample)
        sample = f([mean_vector])[0]
        display_samples(sample)

        assert_array_almost_equal(sample_given_x, sample, decimal=15)


class TestLaplace(TestCase):
    def setUp(self):
        self.seed = 1234567890
        self.mean, self.var = 0, 1
        self.size = 10
        self.mean_sample, self.var_sample = TestDistributionDouble.get_samples(self.mean, self.var)
        self.model = TestLaplace.get_model((1, None), seed=self.seed)
        self.model.set_seed(self.seed)

    @staticmethod
    def get_model(input_size, seed=1):
        mean_layer, var_layer = InputLayer(input_size), InputLayer(input_size)
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

    def test_sample_given_x(self):
        # The output should be the same as GaussianSample.sample when given equals to the input.
        mean_layer = InputLayer((1, None)) # ndim==2
        var_layer = InputLayer((1, None))
        self.model = Laplace(mean_layer, var_layer, given=[mean_layer, var_layer], seed=self.seed)
        self.model.set_seed(self.seed)
        sample_given_x = TestDistributionDouble.get_sample_given_x(
            self.model,
            self.mean_sample,
            self.var_sample
        )
        display_samples(sample_given_x)

        laplace_sample = LaplaceSample(seed=self.seed)
        t_mean = mean_layer.input_var                       # Pass the same theano variable to BernoulliSample.sample
        t_var = var_layer.input_var
        mean_vector = np.ones(self.size).astype("float32") * self.mean
        var_vector = np.ones(self.size).astype("float32") * self.var
        t_sample = laplace_sample.sample(t_mean, t_var)
        f = theano.function(inputs=[t_mean, t_var], outputs=t_sample)
        sample = f([mean_vector], [var_vector])[0]
        display_samples(sample)

        assert_array_almost_equal(sample_given_x, sample, decimal=15)


class TestGamma(TestCase):
    def setUp(self):
        self.seed = 1234567890
        self.alpha, self.beta = 0.5, 0.5
        self.size = 10
        self.alpha_sample, self.beta_sample = TestDistributionDouble.get_samples(self.alpha, self.beta)
        self.model = TestGamma.get_model((1, None), seed=self.seed)
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
            1.1601962737344840e-02,
            1.2021816815566539e+00,
            1.7568047926338815e+00,
            2.1646092000939429e-01,
            4.2426224339589874e-01,
            2.6116683924547507e-01
        ]
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_sample_given_x(self):
        # The output should be the same as GaussianSample.sample when given equals to the input.
        alpha_layer = InputLayer((1, None)) # ndim==2
        beta_layer = InputLayer((1, None))
        self.model = Gamma(alpha_layer, beta_layer, given=[alpha_layer, beta_layer], seed=self.seed)
        self.model.set_seed(self.seed)
        sample_given_x = TestDistributionDouble.get_sample_given_x(
            self.model,
            self.alpha_sample,
            self.beta_sample
        )
        display_samples(sample_given_x)

        gamma_sample = GammaSample(seed=self.seed)
        t_alpha = alpha_layer.input_var                       # Pass the same theano betaiable to BernoulliSample.sample
        t_beta = beta_layer.input_var
        alpha_vector = np.ones(self.size).astype("float32") * self.alpha
        beta_vector = np.ones(self.size).astype("float32") * self.beta
        t_sample = gamma_sample.sample(t_alpha, t_beta)
        f = theano.function(inputs=[t_alpha, t_beta], outputs=t_sample)
        sample = f([alpha_vector], [beta_vector])[0]
        display_samples(sample)

        assert_array_almost_equal(sample_given_x, sample, decimal=15)


class TestBeta(TestCase):
    def setUp(self):
        self.seed = 1234567890
        self.alpha, self.beta = 0.5, 0.5
        self.size = 10
        self.alpha_sample, self.beta_sample = TestDistributionDouble.get_samples(self.alpha, self.beta)
        self.model = TestBeta.get_model((1, None), seed=self.seed)
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
            7.0835564247564431e-03,
            9.2395593282207600e-01,
            5.1113276073478231e-01,
            8.0123533933310609e-01,
            3.0581190019222942e-01,
            9.8989224450713886e-01
        ]
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_sample_given_x(self):
        # The output should be the same as GaussianSample.sample when given equals to the input.
        alpha_layer = InputLayer((1, None)) # ndim==2
        beta_layer = InputLayer((1, None))
        self.model = Beta(alpha_layer, beta_layer, given=[alpha_layer, beta_layer], seed=self.seed)
        self.model.set_seed(self.seed)
        sample_given_x = TestDistributionDouble.get_sample_given_x(
            self.model,
            self.alpha_sample,
            self.beta_sample
        )
        display_samples(sample_given_x)

        beta_dist_sample = BetaSample(seed=self.seed)
        t_alpha = alpha_layer.input_var                       # Pass the same theano betaiable to BernoulliSample.sample
        t_beta = beta_layer.input_var
        alpha_vector = np.ones(self.size).astype("float32") * self.alpha
        beta_vector = np.ones(self.size).astype("float32") * self.beta
        t_sample = beta_dist_sample.sample(t_alpha, t_beta)
        f = theano.function(inputs=[t_alpha, t_beta], outputs=t_sample)
        sample = f([alpha_vector], [beta_vector])[0]
        display_samples(sample)

        assert_array_almost_equal(sample_given_x, sample, decimal=15)


class TestDirichlet(TestCase):
    def setUp(self):
        self.seed = 1234567890
        self.alpha = 0.5
        self.size = 3
        self.k = 3
        self.alpha_sample = TestDistribution.get_samples(self.alpha, self.size)
        self.model = TestDirichlet.get_model((1, self.size), k=self.k, seed=self.seed)
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

    def test_sample_given_x(self):
        # The output should be the same as DirichletSample.sample when given equals to the input.
        # Generate samples from DirichletSample with the same seed as the model
        alpha_layer = InputLayer((1, self.size)) # ndim==2
        self.model = Dirichlet(alpha_layer, given=[alpha_layer], k=self.k, seed=self.seed)
        self.model.set_seed(self.seed)
        sample_given_x = TestDistribution.get_sample_given_x(
            self.model,
            self.alpha_sample,
        )
        display_samples(sample_given_x)

        dirichlet_sample = DirichletSample(self.k, seed=self.seed)
        # Pass the same theano variable to DirichletSample.sample
        t_alpha = alpha_layer.input_var
        alpha_vector = np.ones(self.size).astype("float32") * self.alpha
        t_sample = dirichlet_sample.sample(t_alpha)
        f = theano.function(inputs=[t_alpha], outputs=t_sample)
        sample = f([alpha_vector])[0]
        display_samples(sample)



        assert_array_almost_equal(sample_given_x, sample, decimal=15)


def display_samples(samples, format='e'):
    for i in samples:
        print ('{0:.16%s}'%format).format(i)
    print '='*20

