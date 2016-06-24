from Tars.model import GAN
import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from progressbar import ProgressBar
from ..util import t_repeat, LogMeanExp
from ..distribution import UnitGaussian


class GAN_text(GAN):

    def __init__(self, p, d, n_batch, p_optimizer, d_optimizer, learning_rate=1e-4, beta1=0.5, beta=0.5, random=1234):
        self.p = p
        self.d = d
        self.n_batch = n_batch
        self.p_optimizer = p_optimizer
        self.d_optimizer = d_optimizer
        self.beta = beta

        self.p_sample_mean_given_x()
        
        z = self.p.inputs
        x = self.d.inputs
        p_loss, d_loss, _p_loss = self.loss(z,x)

        p_updates = self.p_optimizer(p_loss+_p_loss, self.p.get_params(), learning_rate=learning_rate, beta1=beta1)
        d_updates = self.d_optimizer(d_loss, self.d.get_params(), learning_rate=learning_rate, beta1=beta1)

        self.p_train = theano.function(inputs=z[:1]+x+z[-1:], outputs=[p_loss,d_loss], updates=p_updates, on_unused_input='ignore')
        self.d_train = theano.function(inputs=z[:1]+x, outputs=[p_loss,d_loss], updates=d_updates, on_unused_input='ignore')

        p_loss, d_loss = self.loss(z,x,True)
        self.test = theano.function(inputs=z[:1]+x, outputs=[p_loss,d_loss], on_unused_input='ignore')

    def loss(self, z, x, deterministic=False):
        p_loss, d_loss = super(GAN_text, self).loss(z[:-1], x)

        _y = z[1]*self.beta + z[2]*(1-self.beta)

        gx = self.p.sample_mean_given_x(z[:1]+[_y], deterministic=deterministic)[-1] # x~p(x|z,_y)
        gt = self.d.sample_mean_given_x([gx]+[_y], deterministic=deterministic)[-1] # gt~d(t|gx,_y)
        _p_loss = -self.d.log_likelihood(T.ones_like(gt),gt).mean() # -log(gt)

        return p_loss, d_loss, _p_loss

    def train(self,train_set, n_z, rng, freq=1):
        pass
