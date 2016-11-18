import numpy as np
import theano
import theano.tensor as T
from progressbar import ProgressBar


class GAN(object):

    def __init__(self, p, d, n_batch, p_optimizer, d_optimizer,
                 learning_rate=1e-4, beta1=0.5, random=1234):
        self.p = p
        self.d = d
        self.n_batch = n_batch
        self.p_optimizer = p_optimizer
        self.d_optimizer = d_optimizer

        self.p_sample_mean_given_x()

        z = self.p.inputs
        x = self.d.inputs
        p_loss, d_loss = self.loss(z, x)

        p_updates = self.p_optimizer(
            p_loss,
            self.p.get_params(),
            learning_rate=learning_rate,
            beta1=beta1)
        d_updates = self.d_optimizer(
            d_loss,
            self.d.get_params(),
            learning_rate=learning_rate,
            beta1=beta1)

        self.p_train = theano.function(
            inputs=z[:1] + x,
            outputs=[p_loss, d_loss],
            updates=p_updates,
            on_unused_input='ignore')
        self.d_train = theano.function(
            inputs=z[:1] + x,
            outputs=[p_loss, d_loss],
            updates=d_updates, on_unused_input='ignore')

        p_loss, d_loss = self.loss(z, x, True)
        self.test = theano.function(
            inputs=z[:1] + x,
            outputs=[p_loss, d_loss],
            on_unused_input='ignore')

    def loss(self, z, x, deterministic=False):
        # gx~p(x|z,y,...)
        gx = self.p.sample_mean_given_x(
            z, deterministic=deterministic)[-1]
        # t~d(t|x,y,...)
        t = self.d.sample_mean_given_x(
            x, deterministic=deterministic)[-1]
        # gt~d(t|gx,y,...)
        gt = self.d.sample_mean_given_x(
            [gx] + x[1:], deterministic=deterministic)[-1]
        # -log(t)
        d_loss = -self.d.log_likelihood(T.ones_like(t), t).mean()
        # -log(1-gt)
        d_g_loss = -self.d.log_likelihood(T.zeros_like(gt), gt).mean()
        # -log(gt)
        p_loss = -self.d.log_likelihood(T.ones_like(gt), gt).mean()

        d_loss = d_loss + d_g_loss  # -log(t)-log(1-gt)

        return p_loss, d_loss

    def train(self, train_set, n_z, rng, freq=1):
        n_x = train_set[0].shape[0]
        nbatches = n_x // self.n_batch
        train = []

        pbar = ProgressBar(maxval=nbatches).start()
        for i in range(nbatches):
            start = i * self.n_batch
            end = start + self.n_batch

            batch_x = [_x[start:end] for _x in train_set]
            batch_z = rng.uniform(-1., 1.,
                                  size=(len(batch_x[0]), n_z)
                                  ).astype(np.float32)
            batch_zx = [batch_z] + batch_x
            if i % (freq + 1) == 0:
                train_L = self.p_train(*batch_zx)
            else:
                train_L = self.d_train(*batch_zx)
            train.append(np.array(train_L))
            pbar.update(i)

        train = np.mean(train, axis=0)

        return train

    def gan_test(self, test_set, n_z, rng):
        n_x = test_set[0].shape[0]
        nbatches = n_x // self.n_batch
        test = []

        pbar = ProgressBar(maxval=nbatches).start()
        for i in range(nbatches):
            start = i * self.n_batch
            end = start + self.n_batch

            batch_x = [_x[start:end] for _x in test_set]
            batch_z = rng.uniform(-1., 1.,
                                  size=(len(batch_x[0]), n_z)
                                  ).astype(np.float32)
            batch_zx = [batch_z] + batch_x
            test_L = self.test(*batch_zx)
            test.append(np.array(test_L))
            pbar.update(i)

        test = np.mean(test, axis=0)

        return test

    def p_sample_mean_given_x(self):
        x = self.p.inputs
        samples = self.p.sample_mean_given_x(x, deterministic=True)
        self.p_sample_mean_x = theano.function(
            inputs=x, outputs=samples[-1], on_unused_input='ignore')
