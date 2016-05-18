import numpy as np
import theano
import theano.tensor as T
import math

from theano.sandbox.cuda.basic_ops import (as_cuda_ndarray_variable,
                                           host_from_gpu,
                                           gpu_contiguous, HostFromGpu,
                                           gpu_alloc_empty)
from theano.sandbox.cuda.dnn import GpuDnnConvDesc, GpuDnnConv, GpuDnnConvGradI, dnn_conv, dnn_pool
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

c = - 0.5 * math.log(2 * math.pi)


def gaussian_like(x, mean, var):
    _var = var + 0.001  # avoid NaN
    return c - T.log(_var) / 2 - (x - mean)**2 / (2 * _var)

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

    elif x.ndim == 4:
        if num_repeats == 1:
            return x
        else:
            if axis == 0:
                return T.alloc(x.dimshuffle(1, 2, 3, 0, 'x'), x.shape[1], x.shape[2], x.shape[3], x.shape[0], num_repeats)\
                        .reshape((x.shape[1], x.shape[2], x.shape[3], num_repeats * x.shape[0]))\
                        .dimshuffle(3, 0, 1, 2)


def LogSumExp(x, axis=0, keepdims=False):
    x_max = T.max(x, axis=axis, keepdims=keepdims)
    _x_max = T.max(x, axis=axis)
    return T.log(T.sum(T.exp(x - x_max), axis=axis)) + _x_max


def LogMeanExp(x, axis=0, keepdims=False):
    x_max = T.max(x, axis=axis, keepdims=keepdims)
    _x_max = T.max(x, axis=axis)
    return T.log(T.mean(T.exp(x - x_max), axis=axis)) + _x_max
