import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from progressbar import ProgressBar
from ..util import t_repeat, LogMeanExp
from ..distribution import UnitGaussian


class GAN(object):

    def __init__(self, p, d, n_batch, p_optimizer, d_optimizer, random=1234):
        self.p = p
        self.d = d
        self.n_batch = n_batch
        self.p_optimizer = p_optimizer
        self.d_optimizer = d_optimizer

        self.p_sample_mean_given_x()

        self.lowerbound(random)
        self.lowerbound_test(random)

    def lowerbound(self, random):
        np.random.seed(random)
        self.srng = RandomStreams(seed=random)

        z = self.p.inputs
        x = self.d.inputs

        gx, p_params = self.p.sample_mean_given_x(z, deterministic=False) # x~p(x|z,y,...)
        t, d_params = self.d.sample_mean_given_x(x, deterministic=False) # t~d(t|x,y,...)
        gt, _ = self.d.sample_mean_given_x([gx]+x[1:], deterministic=False) # gt~d(t|gx,y,...)

        d_lowerbound = self.d.log_likelihood(T.ones_like(t),t).mean() # log(t)
        d_g_lowerbound = self.d.log_likelihood(T.zeros_like(gt),gt).mean() # log(1-gt)
        p_lowerbound = self.d.log_likelihood(T.ones_like(gt),gt).mean() # log(gt)

        d_lowerbound = d_lowerbound + d_g_lowerbound # log(t)+log(1-gt)

        lowerbound = [p_lowerbound, d_lowerbound]
        p_loss = -lowerbound[0]
        d_loss = -lowerbound[1]

        p_updates = self.p_optimizer(p_loss, p_params, learning_rate=1e-4, beta1=0.5)
        d_updates = self.d_optimizer(d_loss, d_params, learning_rate=1e-4, beta1=0.5)

        self.p_lowerbound_train = theano.function(inputs=x+z, outputs=lowerbound, updates=p_updates, on_unused_input='ignore')
        self.d_lowerbound_train = theano.function(inputs=x+z, outputs=lowerbound, updates=d_updates, on_unused_input='ignore')

    def train(self,train_set, n_z,  rng, freq=1):
        N = train_set[0].shape[0]
        nbatches = N // self.n_batch
        lowerbound_train = []

        for i in range(nbatches):
            start = i * self.n_batch
            end = start + self.n_batch

            x = [_x[start:end] for _x in train_set]
            z = rng.uniform(-1., 1., size=(len(x[0]), n_z)).astype(np.float32)
            z = [z] + x[1:]
            xz = x+z
            if i % (freq+1) == 0:
                train_L = self.p_lowerbound_train(*xz)
            else:
                train_L = self.d_lowerbound_train(*xz)
            lowerbound_train.append(np.array(train_L))
        lowerbound_train = np.mean(lowerbound_train, axis=0)

        return lowerbound_train

    def test(self,test_set, n_z, rng):
        N = test_set[0].shape[0]
        nbatches = N // self.n_batch
        lowerbound_test = []

        for i in range(nbatches):
            start = i * self.n_batch
            end = start + self.n_batch

            x = [_x[start:end] for _x in test_set]
            z = rng.uniform(-1., 1., size=(len(x[0]), n_z)).astype(np.float32)
            z = [z] + x[1:]
            xz = x+z
            test_L = self.lowerbound_test(*xz)
            lowerbound_test.append(np.array(test_L))
        lowerbound_test = np.mean(lowerbound_test, axis=0)

        return lowerbound_test


    def p_sample_mean_given_x(self):
        x = self.p.inputs
        samples, _ = self.p.sample_mean_given_x(x, deterministic=True)
        self.p_sample_mean_x = theano.function(
            inputs=x, outputs=samples, on_unused_input='ignore')

    def lowerbound_test(self, random):
        np.random.seed(random)
        self.srng = RandomStreams(seed=random)

        z = self.p.inputs
        x = self.d.inputs

        gx, p_params = self.p.sample_mean_given_x(z, deterministic=True) # x~p(x|z,y,...)
        t, d_params = self.d.sample_mean_given_x(x, deterministic=True) # t~d(t|x,y,...)
        gt, _ = self.d.sample_mean_given_x([gx]+x[1:], deterministic=True) # gt~d(t|gx,y,...)

        d_lowerbound = self.d.log_likelihood(T.ones_like(t),t).mean() # log(t)
        d_g_lowerbound = self.d.log_likelihood(T.zeros_like(gt),gt).mean() # log(1-gt)
        p_lowerbound = self.d.log_likelihood(T.ones_like(gt),gt).mean() # log(gt)

        d_lowerbound = d_lowerbound + d_g_lowerbound # log(t)+log(1-gt)

        lowerbound = [p_lowerbound, d_lowerbound]

        self.lowerbound_test = theano.function(inputs=x+z, outputs=lowerbound, on_unused_input='ignore')
