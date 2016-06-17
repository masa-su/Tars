from Tars.vae._mvae import MVAE
from Tars.vae.gan import GAN
import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from progressbar import ProgressBar
from ..util import t_repeat, LogMeanExp
from ..distribution import UnitGaussian
from copy import copy

class MVAEGAN(MVAE,GAN):

    def __init__(self, q, p, pq, d, n_batch, optimizer, l=1, k=1, random=1234, gan_gamma=1, gamma=1):
        self.d = d
        self.gan_gamma = gan_gamma
        super(MVAEGAN, self).__init__(q, p, pq, n_batch, optimizer, l, k, random)

    def loss(self, gz, x, deterministic=False):
        # TODO: more sophisticated
        _p = self.p
        self.p = self.p[0]
        p_loss, d_loss = super(MVAEGAN, self).loss(gz, x[:1], deterministic)
        self.p = _p

        z = self.q.sample_given_x(x, self.srng, deterministic=deterministic)[-1]
        rec_x = self.p[0].sample_mean_given_x([z], deterministic=deterministic)[-1]
        rec_t = self.d.sample_mean_given_x([rec_x], deterministic=deterministic)[-1] # rec_t~d(rec_t|rec_x,y,...)

        rec_d_g_loss = -self.d.log_likelihood(T.zeros_like(rec_t),rec_t).mean() # -log(1-rec_t)
        rec_p_loss = -self.d.log_likelihood(T.ones_like(rec_t),rec_t).mean() # -log(rec_t)
        p_loss = p_loss + rec_p_loss
        d_loss = d_loss + rec_d_g_loss
        return p_loss, d_loss

    def lowerbound(self):
        x = self.q.inputs
        mean, var = self.q.fprop(x, deterministic=False)
        KL = 0.5 * T.mean(T.sum(1 + T.log(var) - mean**2 - var, axis=1))
        rep_x = [t_repeat(_x, self.l, axis=0) for _x in x]
        z = self.q.sample_given_x(rep_x, self.srng, deterministic=False)

        inverse_z = self.inverse_samples(self.single_input(z,0))
        loglike0 = self.p[0].log_likelihood_given_x(inverse_z)
        loglike0 = T.mean(loglike0)

        inverse_z = self.inverse_samples(self.single_input(z,1))
        loglike1 = self.p[1].log_likelihood_given_x(inverse_z)
        loglike1 = T.mean(loglike1)
        
        # ---penalty
        mean, var = self.q.fprop(x, deterministic=False)
        # z ~ q(x0)
        mean0, var0 = self.pq[0].fprop([x[0]], self.srng, deterministic=False)
        mean1, var1 = self.pq[1].fprop([x[1]], self.srng, deterministic=False)

        # KL[q(x0,0)||q(x0,x1)]
        KL_0  =  self.measure_KL(mean,var,mean0,var0)
        KL_1  =  self.measure_KL(mean,var,mean1,var1)

        # ---GAN---
        gz = self.p[0].inputs
        p_loss, d_loss = self.loss(gz, x, False)

        q_params = self.q.get_params()
        p0_params = self.p[0].get_params()
        p1_params = self.p[1].get_params()
        pq0_params = self.pq[0].get_params()
        pq1_params = self.pq[1].get_params()
        d_params = self.d.get_params()

        lowerbound = [KL, loglike0, loglike1, KL_0, KL_1, p_loss, d_loss]

        q_updates = self.optimizer(-KL-loglike0-loglike1+self.gamma*(KL_0+KL_1), q_params+p1_params+pq0_params+pq1_params, learning_rate=1e-4, beta1=0.5)
        p_updates = self.optimizer(-self.gan_gamma*loglike0 + p_loss, p0_params, learning_rate=1e-4, beta1=0.5)
        d_updates = self.optimizer(d_loss, d_params, learning_rate=1e-4, beta1=0.5)

        self.q_lowerbound_train = theano.function(
            inputs=gz[:1]+x, outputs=lowerbound, updates=q_updates, on_unused_input='ignore')
        self.p_lowerbound_train = theano.function(
            inputs=gz[:1]+x, outputs=lowerbound, updates=p_updates, on_unused_input='ignore')
        self.d_lowerbound_train = theano.function(
            inputs=gz[:1]+x, outputs=lowerbound, updates=d_updates, on_unused_input='ignore')

        p_loss, d_loss = self.loss(gz, x, True)
        self.test = theano.function(inputs=gz[:1]+x, outputs=[p_loss,d_loss], on_unused_input='ignore')

    def train(self, train_set, n_z, rng):
        N = train_set[0].shape[0]
        nbatches = N // self.n_batch
        lowerbound_train = []

        pbar = ProgressBar(maxval=nbatches).start()
        for i in range(nbatches):
            start = i * self.n_batch
            end = start + self.n_batch

            x = [_x[start:end] for _x in train_set]
            z = rng.uniform(-1., 1., size=(len(x[0]), n_z)).astype(np.float32)
            zx = [z]+x

            train_L = self.q_lowerbound_train(*zx)
            train_L = self.p_lowerbound_train(*zx)
            train_L = self.d_lowerbound_train(*zx)
            lowerbound_train.append(np.array(train_L))
            pbar.update(i)

        lowerbound_train = np.mean(lowerbound_train, axis=0)

        return lowerbound_train
