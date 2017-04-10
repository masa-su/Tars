import numpy as np
import theano
import theano.tensor as T
import lasagne
from progressbar import ProgressBar

from ..models.model import Model
from ..distributions.distribution_samples import mean_sum_samples
from ..utils import epsilon


class GAN(Model):

    def __init__(self, p, d, n_batch=100,
                 p_optimizer=lasagne.updates.adam,
                 d_optimizer=lasagne.updates.adam,
                 p_optimizer_params={},
                 d_optimizer_params={},
                 clip_grad=None, max_norm_constraint=None,
                 l1_lambda=0, seed=1234):
        super(GAN, self).__init__(n_batch=n_batch, seed=seed)

        self.p = p
        self.d = d
        self.hidden_dim = self.p.get_input_shape()[0][1:]

        self.l1_lambda = l1_lambda  # for pix2pix

        # set inputs
        z = self.p.inputs
        x = self.d.inputs

        # training
        inputs = z[:1] + x
        loss, params = self._loss(z, x, False)

        p_updates = self._get_updates(loss[0], params[0],
                                      p_optimizer, p_optimizer_params,
                                      clip_grad, max_norm_constraint)
        d_updates = self._get_updates(loss[1], params[1],
                                      d_optimizer, d_optimizer_params,
                                      clip_grad, max_norm_constraint)

        self.p_train = theano.function(inputs=inputs, outputs=loss,
                                       updates=p_updates,
                                       on_unused_input='ignore')
        self.d_train = theano.function(inputs=inputs, outputs=loss,
                                       updates=d_updates,
                                       on_unused_input='ignore')

        # test
        inputs = z[:1] + x
        loss, _ = self._loss(z, x, True)
        self.test = theano.function(inputs=inputs, outputs=loss,
                                    on_unused_input='ignore')

    def _critic(self, x, gx, deterministic=False):
        # t~d(t|x,y,...)
        t = self.d.sample_mean_given_x(
            x, deterministic=deterministic)[-1]
        # gt~d(t|gx,y,...)
        gt = self.d.sample_mean_given_x(
            [gx] + x[1:], deterministic=deterministic)[-1]

        # -log(gt)
        p_loss = -T.log(gt+epsilon())
        # -log(t)-log(1-gt)
        d_loss = -T.log(t+epsilon()) - T.log(1-gt+epsilon())

        return mean_sum_samples(p_loss).mean(), mean_sum_samples(d_loss).mean()

    def _loss(self, z, x, deterministic=False):
        # gx~p(x|z,y,...)
        gx = self.p.sample_mean_given_x(
            z, deterministic=deterministic)[-1]

        p_loss, d_loss = self._critic(x, gx, deterministic)

        if deterministic is False and len(z) > 1:
            p_loss += self.l1_lambda *\
                      mean_sum_samples(T.abs_(x[0]-gx)).mean()

        p_params = self.p.get_params()
        d_params = self.d.get_params()

        return [p_loss, d_loss], [p_params, d_params]

    def train(self, train_set, freq=1, verbose=False):
        n_x = len(train_set[0])
        nbatches = n_x // self.n_batch
        z_dim = (self.n_batch,) + self.hidden_dim
        loss_all = []

        if verbose:
            pbar = ProgressBar(maxval=nbatches).start()
        for i in range(nbatches):
            start = i * self.n_batch
            end = start + self.n_batch
            batch_x = [_x[start:end] for _x in train_set]
            batch_z =\
                self.rng.uniform(-1., 1.,
                                 size=z_dim).astype(batch_x[0].dtype)
            _x = [batch_z] + batch_x
            loss = self.p_train(*_x)
            loss = self.d_train(*_x)
            loss_all.append(np.array(loss))

            if verbose:
                pbar.update(i)

        loss_all = np.mean(loss_all, axis=0)
        return loss_all

    def gan_test(self, test_set, n_batch=None, verbose=False):
        if n_batch is None:
            n_batch = self.n_batch

        n_x = test_set[0].shape[0]
        nbatches = n_x // n_batch
        z_dim = (n_batch,) + self.hidden_dim
        loss_all = []

        if verbose:
            pbar = ProgressBar(maxval=nbatches).start()
        for i in range(nbatches):
            start = i * n_batch
            end = start + n_batch
            batch_x = [_x[start:end] for _x in test_set]
            batch_z =\
                self.rng.uniform(-1., 1.,
                                 size=z_dim).astype(batch_x[0].dtype)

            _x = [batch_z] + batch_x
            loss = self.test(*_x)
            loss_all.append(np.array(loss))

            if verbose:
                pbar.update(i)

        loss_all = np.mean(loss_all, axis=0)
        return loss_all
