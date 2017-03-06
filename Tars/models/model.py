import numpy as np
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from lasagne.updates import total_norm_constraint
from abc import ABCMeta, abstractmethod


class Model(object):
    __metaclass__ = ABCMeta

    def __init__(self, n_batch=100, seed=1234):
        self.n_batch = n_batch
        self.set_seed(seed)
        self.prior_mode = "Normal"

    @abstractmethod
    def train(self):
        pass

    def set_seed(self, seed=1234):
        self.rng = np.random.RandomState(seed)
        self.srng = RandomStreams(seed)

    def _get_updates(self, loss, params, optimizer, optimizer_params={},
                     clip_grad=None, max_norm_constraint=None):
        grads = T.grad(loss, params)
        if max_norm_constraint:
            grads =\
                total_norm_constraint(grads,
                                      max_norm=max_norm_constraint)
        if clip_grad:
            grads = [T.clip(g, -clip_grad, clip_grad) for g in grads]

        return optimizer(grads, params, **optimizer_params)
