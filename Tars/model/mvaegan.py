import numpy as np
import theano
import theano.tensor as T

from progressbar import ProgressBar
from . import (
    MVAE,
    GAN,
)
from ..utils import (
    gauss_gauss_kl,
    gauss_unitgauss_kl,
    t_repeat,
)


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
        KL = gauss_unitgauss_kl(mean, var).mean()
        rep_x = [t_repeat(_x, self.l, axis=0) for _x in x]
        z = self.q.sample_given_x(rep_x, self.srng, deterministic=False)

        inverse_z = self.inverse_samples(self.single_input(z,0))
        loglike0 = self.p[0].log_likelihood_given_x(inverse_z).mean()

        inverse_z = self.inverse_samples(self.single_input(z,1))
        loglike1 = self.p[1].log_likelihood_given_x(inverse_z).mean()
        
        # ---penalty
        mean, var = self.q.fprop(x, deterministic=False)
        # z ~ q(x0)
        mean0, var0 = self.pq[0].fprop([x[0]], self.srng, deterministic=False)
        mean1, var1 = self.pq[1].fprop([x[1]], self.srng, deterministic=False)

        # KL[q(x0,0)||q(x0,x1)]
        KL_0 = gauss_gauss_kl(mean, var, mean0, var0).mean()
        KL_1 = gauss_gauss_kl(mean, var, mean1, var1).mean()

        # ---GAN---
        gz = self.p[0].inputs
        p_loss, d_loss = self.loss(gz, x, False)

        q_params = self.q.get_params()
        p0_params = self.p[0].get_params()
        p1_params = self.p[1].get_params()
        pq0_params = self.pq[0].get_params()
        pq1_params = self.pq[1].get_params()
        d_params = self.d.get_params()

        lowerbound = [-KL, loglike0, loglike1, KL_0, KL_1, p_loss, d_loss]

        q_updates = self.optimizer(KL-loglike0-loglike1+self.gamma*(KL_0+KL_1), q_params+p1_params+pq0_params+pq1_params, learning_rate=1e-4, beta1=0.5)
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

    def train(self, train_set, z_dim, rng):
        n_x = train_set[0].shape[0]
        nbatches = n_x // self.n_batch
        lowerbound_train = []

        pbar = ProgressBar(maxval=nbatches).start()
        for i in range(nbatches):
            start = i * self.n_batch
            end = start + self.n_batch

            batch_x = [_x[start:end] for _x in train_set]
            batch_z = rng.uniform(-1., 1., size=(len(batch_x[0]), z_dim)).astype(np.float32)
            batch_zx = [batch_z]+batch_x

            train_L = self.q_lowerbound_train(*batch_zx)
            train_L = self.p_lowerbound_train(*batch_zx)
            train_L = self.d_lowerbound_train(*batch_zx)
            lowerbound_train.append(np.array(train_L))
            pbar.update(i)

        lowerbound_train = np.mean(lowerbound_train, axis=0)

        return lowerbound_train
