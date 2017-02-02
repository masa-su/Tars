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

    @abstractmethod
    def train(self):
        pass

    def set_seed(self, seed=1234):
        self.rng = np.random.RandomState(seed)
        self.srng = RandomStreams(seed)

    def _inverse_samples(self, samples, prior_mode="Normal", return_prior=False):
        """
        inputs : [[x,y],z1,z2,...zn]
        outputs : p_samples, prior_samples
           if mode is "Normal" : [[zn,y],zn-1,...x], zn
           elif mode is "MultiPrior" : [z1, x], [[zn,y],zn-1,...z1]
        """
        inverse_samples = samples[::-1]
        inverse_samples[0] = [inverse_samples[0]] + inverse_samples[-1][1:]
        inverse_samples[-1] = inverse_samples[-1][0]

        if prior_mode == "Normal":
            p_samples = inverse_samples
            prior_samples = samples[-1]

        elif prior_mode == "MultiPrior":
            p_samples = inverse_samples[-2:]
            prior_samples = inverse_samples[:-1]

        else:
            raise Exception("You should set prior_mode to 'Normal' or"
                            "'MultiPrior', got %s." % prior_mode)

        if prior:
            return p_samples, prior_samples
        else:
            return p_samples

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
