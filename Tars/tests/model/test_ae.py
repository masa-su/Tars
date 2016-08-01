from unittest import TestCase

import numpy as np
from lasagne.layers import InputLayer, DenseLayer
from lasagne.nonlinearities import rectify, linear, sigmoid
from lasagne.updates import adam

from ...distribution import Bernoulli, GaussianConstantVar


class TestAE(TestCase):

    def setUp(self):
        self.nx = 10
        self.nz = 5
        self.layer_dim = 30
        x = InputLayer((None, self.nx))
        q_0 = DenseLayer(x, num_units=self.layer_dim, nonlinearity=rectify)
        q_mean = DenseLayer(q_0, num_units=self.nz, nonlinearity=linear)
        self.q = GaussianConstantVar(q_mean, given=[x])
        z = InputLayer((None, self.nz))
        p_0 = DenseLayer(z, num_units=self.layer_dim, nonlinearity=rectify)
        p_mean = DenseLayer(p_0, num_units=self.nx, nonlinearity=sigmoid)
        self.p = Bernoulli(p_mean, given=[z])

    def test_it(self):
        from ...model import AE
        batch_size = 30
        random_seed = 1234
        sample_size = 100
        model = AE(self.q, self.p, batch_size, adam, random_seed)
        # test p sample mean
        sample_z = np.random.standard_normal(
            (sample_size, self.nz)).astype(np.float32)
        self.assertEqual(
            model.p_sample_mean_x(sample_z).shape,
            (sample_size, self.nx))
        # test q sample mean
        sample_x = np.random.standard_normal(
            (sample_size, self.nx)).astype(np.float32)
        self.assertEqual(
            model.q_sample_mean_x(sample_x).shape,
            (sample_size, self.nz))
        # test train
        train_x = np.random.random((sample_size, self.nx))
        model.train([train_x])
        # test log_marginal_likelihood
        self.assertEqual(
            model.log_marginal_likelihood(train_x[0]).eval().shape,
            (self.nx, )
        )
        # test log_likelihood_test
        all_loglikes = model.log_likelihood_test([train_x])
        self.assertEqual(all_loglikes.shape,
                         (sample_size//batch_size * batch_size, ))
