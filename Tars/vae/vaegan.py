from Tars.vae.vae import VAE
from Tars.vae.gan import GAN
import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from progressbar import ProgressBar
from ..util import t_repeat, LogMeanExp
from ..distribution import UnitGaussian


class VAEGAN(VAE, GAN):

    def __init__(self, q, p, d, n_batch, optimizer, l=1, k=1, gamma=1, random=1234):
        self.d = d
        self.gamma = gamma

        super(VAEGAN, self).__init__(q, p, n_batch, optimizer, l, k, alpha=None, random=random)

    def vaegan_loss(self, gz, x, deterministic=False):
        p_loss, d_loss = self.loss(gz,x,deterministic)

        z = self.q.sample_given_x(x, self.srng, deterministic=deterministic)[-1]
        rec_x = self.p.sample_mean_given_x([z]+x[1:], deterministic=deterministic)[-1]
        rec_t = self.d.sample_mean_given_x([rec_x]+x[1:], deterministic=deterministic)[-1] # rec_t~d(rec_t|rec_x,y,...)

        rec_d_g_loss = -self.d.log_likelihood(T.zeros_like(rec_t),rec_t).mean() # -log(1-rec_t)
        rec_p_loss = -self.d.log_likelihood(T.ones_like(rec_t),rec_t).mean() # -log(rec_t)
        p_loss = p_loss + rec_p_loss
        d_loss = d_loss + rec_d_g_loss
        return p_loss, d_loss

    def lowerbound(self):
        # ---VAE---
        x = self.q.inputs
        mean, var = self.q.fprop(x, deterministic=False)
        KL = 0.5 * T.mean(T.sum(1 + T.log(var) - mean**2 - var, axis=1))
        rep_x = [t_repeat(_x, self.l, axis=0) for _x in x]
        z = self.q.sample_given_x(rep_x, self.srng, deterministic=False)

        inverse_z = self.inverse_samples(z)
        loglike = self.p.log_likelihood_given_x(inverse_z)
        loglike = T.mean(loglike)
        # TODO: feature-wise errors

        # ---GAN---
        gz = self.p.inputs
        p_loss, d_loss = self.vaegan_loss(gz,x,False)

        lowerbound = [KL, loglike, p_loss, d_loss]

        q_params = self.q.get_params()
        p_params = self.p.get_params()
        d_params = self.d.get_params()
        q_updates = self.optimizer(-lowerbound[0] - lowerbound[1], q_params, learning_rate=1e-4, beta1=0.5)
        p_updates = self.optimizer(-self.gamma*lowerbound[1] + p_loss, p_params, learning_rate=1e-4, beta1=0.5)
        d_updates = self.optimizer(d_loss, d_params, learning_rate=1e-4, beta1=0.5)

        self.q_lowerbound_train = theano.function(
            inputs=gz[:1]+x, outputs=lowerbound, updates=q_updates, on_unused_input='ignore')
        self.p_lowerbound_train = theano.function(
            inputs=gz[:1]+x, outputs=lowerbound, updates=p_updates, on_unused_input='ignore')
        self.d_lowerbound_train = theano.function(
            inputs=gz[:1]+x, outputs=lowerbound, updates=d_updates, on_unused_input='ignore')

        p_loss, d_loss = self.vaegan_loss(gz, x, True)
        self.test = theano.function(inputs=gz[:1]+x, outputs=[p_loss,d_loss], on_unused_input='ignore')

    def train(self, train_set, n_z, rng):
        N = train_set[0].shape[0]
        nbatches = N // self.n_batch
        lowerbound_train = []

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
        lowerbound_train = np.mean(lowerbound_train, axis=0)

        return lowerbound_train
