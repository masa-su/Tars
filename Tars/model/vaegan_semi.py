from Tars.model.vaegan import VAEGAN
import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from progressbar import ProgressBar
from ..util import KL_gauss_unitgauss, t_repeat, LogMeanExp
from ..distribution import UnitGaussian


class VAEGAN_semi(VAEGAN):

    def __init__(self, q, p, d, f, n_batch, optimizer, l=1, k=1, gamma=1, f_alpha=0.1, random=1234):
        self.f = f
        self.f_alpha = f_alpha
        super(VAEGAN_semi, self).__init__(q, p, d, n_batch, optimizer, l, k, gamma, random=random)
        self.f_sample_mean_given_x()

    def lowerbound(self):
        # ---VAE---
        x = self.q.inputs
        mean, var = self.q.fprop(x, deterministic=False)
        KL = KL_gauss_unitgauss(mean, var).mean()
        rep_x = [t_repeat(_x, self.l, axis=0) for _x in x]
        z = self.q.sample_given_x(rep_x, self.srng, deterministic=False)

        inverse_z = self.inverse_samples(z)
        loglike = self.p.log_likelihood_given_x(inverse_z).mean()
        # TODO: feature-wise errors

        # --semi_supervise
        x_unlabel = self.f.inputs
        y = self.f.sample_mean_given_x(x_unlabel, self.srng, deterministic=False)[-1]
        mean, var = self.q.fprop([x_unlabel[0],y], self.srng, deterministic=False)
        KL_semi = KL_gauss_unitgauss(mean, var).mean()

        rep_x_unlabel = [t_repeat(_x, self.l, axis=0) for _x in x_unlabel]
        rep_y = self.f.sample_mean_given_x(rep_x_unlabel, self.srng, deterministic=False)[-1]
        z = self.q.sample_given_x([rep_x_unlabel[0],rep_y], self.srng, deterministic=False)      
        inverse_z = self.inverse_samples(z)
        loglike_semi = self.p.log_likelihood_given_x(inverse_z).mean()

        # --train f
        loglike_f = self.f.log_likelihood_given_x([[x[0]],x[1]]).mean()

        # ---GAN---
        gz = self.p.inputs
        p_loss, d_loss = self.loss(gz,x,False)

        lowerbound = [-KL, loglike, p_loss, d_loss, -KL_semi, loglike_semi, loglike_f]

        q_params = self.q.get_params()
        p_params = self.p.get_params()
        d_params = self.d.get_params()
        f_params = self.f.get_params()

        q_updates = self.optimizer(KL -loglike +KL_semi -loglike_semi -self.f_alpha * loglike_f, q_params+f_params, learning_rate=1e-4, beta1=0.5)
        p_updates = self.optimizer(-self.gamma*(loglike + loglike_semi) + p_loss, p_params, learning_rate=1e-4, beta1=0.5)
        d_updates = self.optimizer(d_loss, d_params, learning_rate=1e-4, beta1=0.5)

        self.q_lowerbound_train = theano.function(
            inputs=gz[:1]+x+x_unlabel, outputs=lowerbound, updates=q_updates, on_unused_input='ignore')
        self.p_lowerbound_train = theano.function(
            inputs=gz[:1]+x+x_unlabel, outputs=lowerbound, updates=p_updates, on_unused_input='ignore')
        self.d_lowerbound_train = theano.function(
            inputs=gz[:1]+x+x_unlabel, outputs=lowerbound, updates=d_updates, on_unused_input='ignore')

        p_loss, d_loss = self.loss(gz, x, True)
        self.test = theano.function(inputs=gz[:1]+x, outputs=[p_loss,d_loss], on_unused_input='ignore')

    def train(self, train_set, train_set_unlabel, z_dim, rng):
        n_x = train_set[0].shape[0]
        nbatches = n_x // self.n_batch
        lowerbound_train = []

        n_x_unlabel = train_set_unlabel[0].shape[0]
        n_batch_unlabel = n_x_unlabel // nbatches

        pbar = ProgressBar(maxval=nbatches).start()
        for i in range(nbatches):
            start = i * self.n_batch
            end = start + self.n_batch

            batch_x = [_x[start:end] for _x in train_set]
            batch_z = rng.uniform(-1., 1., size=(len(batch_x[0]), z_dim)).astype(np.float32)
            batch_zx = [batch_z]+batch_x

            start = i * n_batch_unlabel
            end = start + n_batch_unlabel
            batch_x_unlabel = [_x[start:end] for _x in train_set_unlabel]

            train_L = self.q_lowerbound_train(*batch_zx+batch_x_unlabel)
            train_L = self.p_lowerbound_train(*batch_zx+batch_x_unlabel)
            train_L = self.d_lowerbound_train(*batch_zx+batch_x_unlabel)
            lowerbound_train.append(np.array(train_L))
            pbar.update(i)

        lowerbound_train = np.mean(lowerbound_train, axis=0)

        return lowerbound_train

    def f_sample_mean_given_x(self):
        x = self.f.inputs
        samples = self.f.sample_mean_given_x(x, self.srng, deterministic=True)
        self.f_sample_mean_x = theano.function(
            inputs=x, outputs=samples[-1], on_unused_input='ignore')
