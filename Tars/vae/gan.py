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

        z = self.p.inputs
        x = self.d.inputs
        p_loss, d_loss, p_param, d_param = self.loss(z,x)
        p_updates = self.p_optimizer(p_loss, p_param, learning_rate=1e-4, beta1=0.5)
        d_updates = self.d_optimizer(d_loss, d_param, learning_rate=1e-4, beta1=0.5)

        self.p_train = theano.function(inputs=x+z, outputs=[p_loss,d_loss], updates=p_updates, on_unused_input='ignore')
        self.d_train = theano.function(inputs=x+z, outputs=[p_loss,d_loss], updates=d_updates, on_unused_input='ignore')

        p_loss, d_loss, _, _ = self.loss(z,x,True)
        self.test = theano.function(inputs=x+z, outputs=[p_loss,d_loss], on_unused_input='ignore')

    def loss(self, z, x, deterministic=False):
        gx, p_param = self.p.sample_mean_given_x(z, deterministic=deterministic) # x~p(x|z,y,...)
        t, d_param = self.d.sample_mean_given_x(x, deterministic=deterministic) # t~d(t|x,y,...)
        gt, _ = self.d.sample_mean_given_x([gx]+x[1:], deterministic=deterministic) # gt~d(t|gx,y,...)

        d_loss = -self.d.log_likelihood(T.ones_like(t),t).mean() # -log(t)
        d_g_loss = -self.d.log_likelihood(T.zeros_like(gt),gt).mean() # -log(1-gt)
        p_loss = -self.d.log_likelihood(T.ones_like(gt),gt).mean() # -log(gt)

        d_loss = d_loss + d_g_loss # -log(t)-log(1-gt)

        return p_loss, d_loss, p_param, d_param

    def train(self,train_set, n_z,  rng, freq=1):
        N = train_set[0].shape[0]
        nbatches = N // self.n_batch
        train = []

        for i in range(nbatches):
            start = i * self.n_batch
            end = start + self.n_batch

            x = [_x[start:end] for _x in train_set]
            z = rng.uniform(-1., 1., size=(len(x[0]), n_z)).astype(np.float32)
            z = [z] + x[1:]
            xz = x+z
            if i % (freq+1) == 0:
                train_L = self.p_train(*xz)
            else:
                train_L = self.d_train(*xz)
            train.append(np.array(train_L))
        train = np.mean(train, axis=0)

        return train

    def gan_test(self,test_set, n_z, rng):
        N = test_set[0].shape[0]
        nbatches = N // self.n_batch
        test = []

        for i in range(nbatches):
            start = i * self.n_batch
            end = start + self.n_batch

            x = [_x[start:end] for _x in test_set]
            z = rng.uniform(-1., 1., size=(len(x[0]), n_z)).astype(np.float32)
            z = [z] + x[1:]
            xz = x+z
            test_L = self.test(*xz)
            test.append(np.array(test_L))
        test = np.mean(test, axis=0)

        return test

    def p_sample_mean_given_x(self):
        x = self.p.inputs
        samples, _ = self.p.sample_mean_given_x(x, deterministic=True)
        self.p_sample_mean_x = theano.function(
            inputs=x, outputs=samples, on_unused_input='ignore')
