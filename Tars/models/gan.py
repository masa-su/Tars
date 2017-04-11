import numpy as np
import theano
import theano.tensor as T
import lasagne
from progressbar import ProgressBar

from ..models.model import Model
from ..distributions.distribution_samples import UnitUniformSample, mean_sum_samples
from ..utils import epsilon


class GAN(Model):

    def __init__(self, p, d, prior=UnitUniformSample(),
                 n_batch=100,
                 p_optimizer=lasagne.updates.adam,
                 d_optimizer=lasagne.updates.adam,
                 p_optimizer_params={},
                 d_optimizer_params={},
                 p_critic=lambda gt: -T.log(gt+epsilon()),
                 d_critic=lambda t, gt: -T.log(t+epsilon()) - T.log(1-gt+epsilon()),
                 p_clip_param=None, d_clip_param=None,
                 p_clip_grad=None, d_clip_grad=None,
                 p_max_norm_constraint=None,
                 d_max_norm_constraint=None,
                 l1_lambda=0, seed=1234):
        super(GAN, self).__init__(n_batch=n_batch, seed=seed)

        self.p = p
        self.d = d
        self.prior = prior

        self.l1_lambda = l1_lambda  # for pix2pix

        # Set inputs
        x = self.d.inputs  # x=[x,y,...]

        # Check inputs are correct.
        # We assume d.inputs=[x,y,...] and p_inputs=[y,...] or
        # d.inputs=[x,y,...] and p_inputs=[z,y,...]
        if prior is None:
            p_inputs = self.p.inputs
        else:
            p_inputs = self.p.inputs[1:]
        if x[1:] != p_inputs:
            raise ValueError('You should set same conditional variables on '
                             'both the generator and discriminator')
        self.z_shape = (x[0].shape[0],) + self.p.get_input_shape()[0][1:]

        # set critic
        self.p_critic = p_critic
        self.d_critic = d_critic

        # training
        loss, params = self._loss(x, False)

        p_updates = self._get_updates(loss[0], params[0],
                                      p_optimizer, p_optimizer_params,
                                      p_clip_param, p_clip_grad,
                                      p_max_norm_constraint)
        d_updates = self._get_updates(loss[1], params[1],
                                      d_optimizer, d_optimizer_params,
                                      d_clip_param, d_clip_grad,
                                      d_max_norm_constraint)

        self.p_train = theano.function(inputs=x, outputs=loss,
                                       updates=p_updates,
                                       on_unused_input='ignore')
        self.d_train = theano.function(inputs=x, outputs=loss,
                                       updates=d_updates,
                                       on_unused_input='ignore')

        # test
        loss, _ = self._loss(x, True)
        self.test = theano.function(inputs=x, outputs=loss,
                                    on_unused_input='ignore')

    def _loss(self, x, deterministic=False):
        # z~p(z)
        z = []
        if self.prior is not None:
            z.append(self.prior.sample(self.z_shape))

        # gx~p(x|z,y,...)
        gx = self.p.sample_mean_given_x(
            z+x[1:], deterministic=deterministic)[-1]

        # t~d(t|x,y,...)
        t = self.d.sample_mean_given_x(
            x, deterministic=deterministic)[-1]
        # gt~d(t|gx,y,...)
        gt = self.d.sample_mean_given_x(
            [gx] + x[1:], deterministic=deterministic)[-1]

        p_loss = mean_sum_samples(self.p_critic(gt)).mean()
        d_loss = mean_sum_samples(self.d_critic(t, gt)).mean()

        if deterministic is False and len(z) > 1:
            p_loss +=\
                self.l1_lambda * mean_sum_samples(T.abs_(x[0]-gx)).mean()

        p_params = self.p.get_params()
        d_params = self.d.get_params()

        return [p_loss, d_loss], [p_params, d_params]

    def train(self, train_set, freq=1, verbose=False):
        n_x = len(train_set[0])
        nbatches = n_x // self.n_batch
        loss_all = []

        if verbose:
            pbar = ProgressBar(maxval=nbatches).start()
        for i in range(nbatches):
            start = i * self.n_batch
            end = start + self.n_batch
            batch_x = [_x[start:end] for _x in train_set]

            for _ in range(freq):
                loss = self.d_train(*batch_x)
            loss = self.p_train(*batch_x)
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
        loss_all = []

        if verbose:
            pbar = ProgressBar(maxval=nbatches).start()
        for i in range(nbatches):
            start = i * n_batch
            end = start + n_batch
            batch_x = [_x[start:end] for _x in test_set]

            loss = self.test(*batch_x)
            loss_all.append(np.array(loss))

            if verbose:
                pbar.update(i)

        loss_all = np.mean(loss_all, axis=0)
        return loss_all
