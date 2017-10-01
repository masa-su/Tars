from copy import copy, deepcopy

import theano
import theano.tensor as T
import lasagne
import numpy as np
from progressbar import ProgressBar

from ..utils import tolist, log_mean_exp
from ..distributions.estimate_kl import analytical_kl
from ss_hmvae import SS_HMVAE


class SS_HMVAE_KL(SS_HMVAE):
    def __init__(self, q, p, f, s_q, pseudo_q,
                 prior=None, gamma=1.0,
                 n_batch=100, n_batch_u=100,
                 optimizer=lasagne.updates.adam,
                 optimizer_params={},
                 sum_classes=False,
                 clip_grad=None, max_norm_constraint=None,
                 seed=1234):
        self.pseudo_q = pseudo_q
        self.gamma = gamma
        print self.gamma
        super(SS_HMVAE_KL,
              self).__init__(q, p, f, s_q, prior=prior,
                             n_batch=n_batch,
                             optimizer=optimizer,
                             optimizer_params=optimizer_params,
                             sum_classes=sum_classes,
                             clip_grad=clip_grad,
                             max_norm_constraint=max_norm_constraint,
                             seed=seed)

    def _set_train(self, l, k, *args):
        # inputs
        x_u = self.q.inputs
        x_l = deepcopy(self.q.inputs)
        y = T.fmatrix("y")

        # training
        rate = T.fscalar("rate")
        inputs = x_u + x_l + [y, l, k, rate]
        lower_bound_u, loss_u, [params, ss_params] =\
            self._vr_bound(x_u, l, k, 0, False)
        lower_bound_l, loss_l, _ = self._vr_bound(x_l, l, k, 0, False,
                                                  tolist(y))
        lower_bound_y, loss_y, _ = self._discriminate(x_l, tolist(y), False)

        loss_kl_l, kl_params = self._kl_term(x_l, False)
        loss_kl_u, _ = self._kl_term(x_u, False)
        all_params = params + ss_params + kl_params

        # whether delete conflicted params
        params = sorted(set(params), key=params.index)

        lower_bound = [T.mean(lower_bound_u), T.mean(lower_bound_l),
                       T.mean(lower_bound_y)]

        loss = loss_u + loss_l + rate * loss_y + self.gamma * (loss_kl_u + loss_kl_l)
        updates = self._get_updates(loss, all_params, self.optimizer,
                                    self.optimizer_params, self.clip_grad,
                                    self.max_norm_constraint)

        self.lower_bound_train = theano.function(inputs=inputs,
                                                 outputs=lower_bound,
                                                 updates=updates,
                                                 on_unused_input='ignore')

        # training (jmvae part)
        updates = self._get_updates(loss, params+kl_params, self.optimizer,
                                    self.optimizer_params, self.clip_grad,
                                    self.max_norm_constraint)

        self.lower_bound_jmvae_train = theano.function(inputs=inputs,
                                                 outputs=lower_bound,
                                                 updates=updates,
                                                 on_unused_input='ignore')


        # training (semi-supervised part)
        updates = self._get_updates(loss, ss_params, self.optimizer,
                                    self.optimizer_params, self.clip_grad,
                                    self.max_norm_constraint)

        self.lower_bound_ss_train = theano.function(inputs=inputs,
                                                 outputs=lower_bound,
                                                 updates=updates,
                                                 on_unused_input='ignore')

        # training (classification)
        inputs = x_l + [y]
        lower_bound_y, loss, [params, ss_params] =\
            self._discriminate(x_l, tolist(y), False)
        loss_kl_l, kl_params = self._kl_term(x_l, False)
        
        loss = self.gamma * loss_kl_l + loss
        params = params + kl_params

        # whether delete conflicted params
        params = sorted(set(params), key=params.index)

        updates = self._get_updates(loss, params+ss_params, self.optimizer,
                                    self.optimizer_params, self.clip_grad,
                                    self.max_norm_constraint)

        self.classifier_train = theano.function(inputs=inputs,
                                                outputs=T.mean(lower_bound_y),
                                                updates=updates,
                                                on_unused_input='ignore')

    def _kl_term(self, x, deterministic=False):
        kl_all = []
        pseudo_q_params = []
        for i, _pseudo_q in enumerate(self.pseudo_q):
            if self.q.__class__.__name__ == "MultiDistributions":
                q = self.q.distributions[0]
            else:
                q = self.q
            kl_all.append(analytical_kl(q, _pseudo_q, given=[x, [x[i]]],
                                        deterministic=deterministic).mean())
#                                        deterministic=deterministic))
            pseudo_q_params += _pseudo_q.get_params()

#        loss = T.mean(kl_all)
        loss = sum(kl_all)
        params = pseudo_q_params

        return loss, params
    """
    def train(self, train_set_u, train_set_l, l=1, k=1,
              nbatches=2000, get_batch_samples=None, gamma=1,
              discriminate_rate=1, verbose=False, **kwargs):
        lower_bound_all = []

        if verbose:
            pbar = ProgressBar(maxval=nbatches).start()

        _n_batch_u = len(train_set_u[0]) / nbatches
        _n_batch = len(train_set_l[0]) / nbatches

        for i in range(nbatches):
            if get_batch_samples:
                # unlabel
                batch_set_u = get_batch_samples(train_set_u,
                                                n_batch=self.n_batch_u)
                # label
                batch_set_l = get_batch_samples(train_set_l,
                                                n_batch=self.n_batch)
            else:
                # unlabel
                start_u = i * _n_batch_u
                end_u = start_u + _n_batch_u
                batch_set_u = [_x[start_u:end_u] for _x in train_set_u]

                # label
                start = i * _n_batch
                end = start + _n_batch
                batch_set_l = [_x[start:end] for _x in train_set_l]
                
            _x = batch_set_u + batch_set_l + [l, k, discriminate_rate, gamma]
            lower_bound = self.lower_bound_ss_train(*_x)
            lower_bound = self.lower_bound_jmvae_train(*_x)
            lower_bound_all.append(np.array(lower_bound))

            if verbose:
                pbar.update(i)

        lower_bound_all = np.mean(lower_bound_all, axis=0)
        return lower_bound_all

    def train_classifier(self, train_set_l, nbatches=2000,
                         get_batch_samples=None, verbose=False,
                         gamma=1,
                         **kwargs):

        lower_bound_all = []

        if verbose:
            pbar = ProgressBar(maxval=nbatches).start()

        _n_batch = len(train_set_l[0]) / nbatches

        for i in range(nbatches):
            if get_batch_samples:
                # label
                batch_set_l = get_batch_samples(train_set_l,
                                                n_batch=self.n_batch)
            else:
                # label
                start = i * _n_batch
                end = start + _n_batch
                batch_set_l = [_x[start:end] for _x in train_set_l]
            batch_set_l = batch_set_l + [gamma]
            lower_bound = self.classifier_train(*batch_set_l)
            lower_bound_all.append(np.array(lower_bound))

            if verbose:
                pbar.update(i)

        lower_bound_all = np.mean(lower_bound_all, axis=0)
        return lower_bound_all
    """
