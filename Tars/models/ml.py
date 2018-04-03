import numpy as np
from progressbar import ProgressBar
from copy import copy, deepcopy

import theano
import theano.tensor as T
import lasagne

from ..utils import log_mean_exp, tolist
from ..models.model import Model


class ML(Model):

    def __init__(self, f,
                 n_batch=100, optimizer=lasagne.updates.adam,
                 optimizer_params={},
                 clip_grad=None, max_norm_constraint=None,
                 seed=1234):
        super(ML, self).__init__(n_batch=n_batch, seed=seed)
        self.f = f

        # set inputs
        x = self.f.inputs
        y = tolist(T.fmatrix("y"))

        # training
        loss, params = self._loss(x, y, False)
        loss = T.mean(loss, axis=0)
        updates = self._get_updates(-loss, params, optimizer, optimizer_params,
                                    clip_grad, max_norm_constraint)

        inputs = x + y
        self.loss_train = theano.function(inputs=inputs,
                                          outputs=loss,
                                          updates=updates,
                                          on_unused_input='ignore')

        # test
        loss, _ = self._loss(x, y, True)
        self.loss_test = theano.function(inputs=inputs,
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

    def _loss(self, x, y, deterministic=False):
        # log q(y|x)
        # _samples : [[x], y]
        _samples = [x] + y
        log_likelihood = self.f.log_likelihood_given_x(
            _samples, deterministic=deterministic)
        params = self.f.get_params()

        return log_likelihood, params
