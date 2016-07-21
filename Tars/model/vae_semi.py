from Tars.model import VAE
import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from progressbar import ProgressBar
from ..utils import gauss_gauss_kl, gauss_unitgauss_kl, t_repeat, log_mean_exp
from ..distribution import UnitGaussian


class VAE_semi(VAE):

    def __init__(self, q, p, f, n_batch, optimizer, l=1, k=1, f_alpha=0.1, random=1234):
        self.f = f
        self.f_alpha = f_alpha
        super(VAE_semi, self).__init__(q, p, n_batch, optimizer, l, k, alpha=None, random=random)
        self.f_sample_mean_given_x()

    def lowerbound(self):
        x = self.q.inputs
        annealing_beta = T.fscalar("beta")

        mean, var = self.q.fprop(x, self.srng, deterministic=False)
        kl = gauss_unitgauss_kl(mean, var).mean()
        rep_x = [t_repeat(_x, self.l, axis=0) for _x in x]
        z = self.q.sample_given_x(rep_x, self.srng, deterministic=False)
        
        inverse_z = self.inverse_samples(z)
        loglike = self.p.log_likelihood_given_x(inverse_z).mean()

        # --semi_supervise
        x_unlabel = self.f.inputs
        y = self.f.sample_mean_given_x(x_unlabel, self.srng, deterministic=False)[-1]
        mean, var = self.q.fprop([x_unlabel[0],y], self.srng, deterministic=False)
        kl_semi = gauss_unitgauss_kl(mean, var).mean()

        rep_x_unlabel = [t_repeat(_x, self.l, axis=0) for _x in x_unlabel]
        rep_y = self.f.sample_mean_given_x(rep_x_unlabel, self.srng, deterministic=False)[-1]
        z = self.q.sample_given_x([rep_x_unlabel[0],rep_y], self.srng, deterministic=False)      
        inverse_z = self.inverse_samples(z)
        loglike_semi = self.p.log_likelihood_given_x(inverse_z).mean()

        # --train f
        loglike_f = self.f.log_likelihood_given_x([[x[0]],x[1]]).mean()

        lowerbound = [-kl, loglike, -kl_semi, loglike_semi, loglike_f]
        loss = annealing_beta*kl -np.sum(lowerbound[1:-1]) - self.f_alpha*lowerbound[-1]

        q_params = self.q.get_params()
        p_params = self.p.get_params()
        f_params = self.f.get_params()
        params = q_params + p_params + f_params

        updates = self.optimizer(loss, params)
        self.lowerbound_train = theano.function(
            inputs=x+x_unlabel+[annealing_beta], outputs=lowerbound, updates=updates, on_unused_input='ignore')

        self.lowerbound_test = theano.function(
            inputs=x+x_unlabel, outputs=lowerbound, on_unused_input='ignore')

    def train(self, train_set, train_set_unlabel, annealing_beta=1):
        n_x = train_set[0].shape[0]
        nbatches = n_x // self.n_batch
        lowerbound_train = []

        n_x_unlabel = train_set_unlabel[0].shape[0]
        n_batch_unlabel = n_x_unlabel // nbatches

        for i in range(nbatches):
            start = i * self.n_batch
            end = start + self.n_batch
            batch_x = [_x[start:end] for _x in train_set]

            start = i * n_batch_unlabel
            end = start + n_batch_unlabel
            batch_x_unlabel = [_x[start:end] for _x in train_set_unlabel]

            train_L = self.lowerbound_train(*batch_x+batch_x_unlabel+[annealing_beta])
            lowerbound_train.append(np.array(train_L))
        lowerbound_train = np.mean(lowerbound_train, axis=0)

        return lowerbound_train

    def f_sample_mean_given_x(self):
        x = self.f.inputs
        samples = self.f.sample_mean_given_x(x, self.srng, deterministic=True)
        self.f_sample_mean_x = theano.function(
            inputs=x, outputs=samples[-1], on_unused_input='ignore')
