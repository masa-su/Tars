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


def log_sum_exp(x, axis=0, keepdims=False):
    x_max = T.max(x, axis=axis, keepdims=True)
    _x_max = T.max(x, axis=axis, keepdims=keepdims)
    return T.log(T.sum(T.exp(x - x_max), axis=axis, keepdims=keepdims)) + _x_max


def log_mean_exp(x, axis=0, keepdims=False):
    x_max = T.max(x, axis=axis, keepdims=True)
    _x_max = T.max(x, axis=axis, keepdims=keepdims)
    return T.log(T.mean(T.exp(x - x_max), axis=axis, keepdims=keepdims)) + _x_max


def tolist(x):
    if isinstance(x, list):
        return x
    elif isinstance(x, tuple):
        return list(x)
    return [x]
