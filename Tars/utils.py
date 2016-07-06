import math

import lasagne.layers
import numpy as np
import theano.tensor as T

_EPSILON = np.finfo(np.float32).eps


def set_epsilon(eps):
    global _EPSILON
    _EPSILON = eps


def epsilon():
    return _EPSILON


def save_weights(network, name):
    np.savez(name, *lasagne.layers.get_all_param_values(network))


def load_weights(network, name):
    with np.load(name) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(network, param_values)


def gaussian_like(x, mean, var):
    c = - 0.5 * math.log(2 * math.pi)
    _var = var + epsilon()  # avoid NaN
    return c - T.log(_var) / 2 - (x - mean)**2 / (2 * _var)


def gauss_unitgauss_kl(mean, var):
    return -0.5 * T.sum(1 + T.log(var) - mean**2 - var, axis=1)


def gauss_gauss_kl(mean0, var0, mean1, var1):
    kl = T.log(var1) - T.log(var0) + T.exp(T.log(var0) - T.log(var1)) + (mean0 - mean1)**2 / T.exp(T.log(var1))
    return 0.5 * T.sum(kl, axis=1)


# https://github.com/yburda/iwae/blob/master/utils.py
def t_repeat(x, num_repeats, axis):
    '''Repeats x along an axis num_repeats times. Axis has to be 0 or 1, x has to be a matrix.'''
    if x.ndim == 2:
        if num_repeats == 1:
            return x
        else:
            if axis == 0:
                return T.alloc(x.dimshuffle(1, 0, 'x'), x.shape[1], x.shape[0], num_repeats)\
                        .reshape((x.shape[1], num_repeats * x.shape[0]))\
                        .dimshuffle(1, 0)
            elif axis == 1:
                return T.alloc(x.dimshuffle(0, 'x', 1), x.shape[0], num_repeats, x.shape[1]).reshape((x.shape[0], num_repeats * x.shape[1]))

    elif x.ndim == 3:
        if num_repeats == 1:
            return x
        else:
            if axis == 0:
                return T.alloc(x.dimshuffle(1, 2, 0, 'x'), x.shape[1], x.shape[2], x.shape[0], num_repeats)\
                        .reshape((x.shape[1], x.shape[2], num_repeats * x.shape[0]))\
                        .dimshuffle(2, 0, 1)

    elif x.ndim == 4:
        if num_repeats == 1:
            return x
        else:
            if axis == 0:
                return T.alloc(x.dimshuffle(1, 2, 3, 0, 'x'), x.shape[1], x.shape[2], x.shape[3], x.shape[0], num_repeats)\
                        .reshape((x.shape[1], x.shape[2], x.shape[3], num_repeats * x.shape[0]))\
                        .dimshuffle(3, 0, 1, 2)


def log_sum_exp(x, axis=0, keepdims=False):
    x_max = T.max(x, axis=axis, keepdims=keepdims)
    _x_max = T.max(x, axis=axis)
    return T.log(T.sum(T.exp(x - x_max), axis=axis)) + _x_max


def log_mean_exp(x, axis=0, keepdims=False):
    x_max = T.max(x, axis=axis, keepdims=keepdims)
    _x_max = T.max(x, axis=axis)
    return T.log(T.mean(T.exp(x - x_max), axis=axis)) + _x_max


def tolist(x):
    if isinstance(x, list):
        return x
    return [x]
