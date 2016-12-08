import numpy as np
import theano
import theano.tensor as T
import lasagne
from progressbar import ProgressBar

from ..model.model import Model


class AE(Model):

    def __init__(self, q, p,
                 n_batch=100, optimizer=lasagne.updates.adam,
                 optimizer_params={},
                 clip_grad=None, max_norm_constraint=None,
                 seed=1234):
        super(AE, self).__init__(n_batch=n_batch, seed=seed)
        self.q = q
        self.p = p

        # set inputs
        x = self.q.inputs

        # training
        loss, params = self._loss(x, False)
        loss = T.mean(loss, axis=0)
        updates = self._get_updates(loss, params, optimizer, optimizer_params,
                                    clip_grad, max_norm_constraint)

        self.loss_train = theano.function(inputs=x,
                                          outputs=loss,
                                          updates=updates,
                                          on_unused_input='ignore')

        # test
        loss, _ = self._loss(x, True)
        self.loss_test = theano.function(inputs=x,
                                         outputs=loss,
                                         on_unused_input='ignore')

    def train(self, train_set, verbose=False):
        n_x = train_set[0].shape[0]
        nbatches = n_x // self.n_batch
        loss_all = []

        if verbose:
            pbar = ProgressBar(maxval=nbatches).start()
        for i in range(nbatches):
            start = i * self.n_batch
            end = start + self.n_batch
            batch_x = [_x[start:end] for _x in train_set]
            loss = self.loss_train(*batch_x)
            loss_all.append(np.array(loss))

            if verbose:
                pbar.update(i)

        loss_all = np.mean(loss_all, axis=0)
        return loss_all

    def test(self, test_set, n_batch=None, verbose=True):
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
            loss = self.loss_test(*batch_x)
            loss_all = np.r_[loss_all, loss]

            if verbose:
                pbar.update(i)

        return loss_all

    def _loss(self, x, deterministic=False):
        z = self.q.sample_mean_given_x(x, deterministic=False)
        inverse_z = self._inverse_samples(z)
        loss = -self.p.log_likelihood_given_x(inverse_z,
                                              deterministic=deterministic)

        q_params = self.q.get_params()
        p_params = self.p.get_params()
        params = q_params + p_params

        return loss, params
