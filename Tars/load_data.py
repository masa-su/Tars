import gzip
import cPickle
import numpy as np
import scipy as sp
from scipy.io import loadmat
import glob
import os
from sklearn import preprocessing
import sys
import pickle
from copy import copy


def one_of_k(a):
    a = np.array(a)
    b = np.zeros((a.size, 10)).astype('float32')
    b[np.arange(a.size), a] = 1
    return b


def mnist(datapath, toFloat=False):
    p = paramaters()

    def load(test=False, one_hot=True):
        f = gzip.open(datapath + 'mnist.pkl.gz', 'rb')
        (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = cPickle.load(f)
        f.close()

        if toFloat:
            p.mean = np.mean(train_x, axis=0)
            p.std = np.sqrt(np.mean((train_x - p.mean[np.newaxis])**2, axis=0))
            p.std[p.std == 0] = 1
            train_x = ((train_x - p.mean[np.newaxis]) /
                       p.std[np.newaxis]).astype(np.float32)
            test_x = ((test_x - p.mean[np.newaxis]) /
                      p.std[np.newaxis]).astype(np.float32)

        if one_hot:
            train_y = one_of_k(train_y)
            valid_y = one_of_k(valid_y)
            test_y = one_of_k(test_y)

        if test:
            return train_x, train_y, valid_x, valid_y, test_x, test_y
        else:
            return train_x, train_y

    def plot(X):
        if toFloat:
            X = (X * p.std[np.newaxis]) + p.mean[np.newaxis]
            X = X / 255.
            X[X < 0] = 0
            X[X > 1] = 1
        X = X.reshape((X.shape[0], 28, 28))
        return 1 - X, "gray"
    return load, plot


class paramaters():

    def __init__(self, mean=0, std=0):
        self.mean = mean
        self.std = std


def svhn(datapath, toFloat=True, binarize_y=True, gray=False, extra=True):
    p = paramaters()

    def load(test=False, flatten=False):
        trains = loadmat(datapath + 'train_32x32.mat')
        tests = loadmat(datapath + 'test_32x32.mat')
        train_x = trains['X'].swapaxes(0, 1).T
        train_y = trains['y'].reshape((-1))
        test_x = tests['X'].swapaxes(0, 1).T
        test_y = tests['y'].reshape((-1))

        if extra:
            extras = loadmat(datapath + 'extra_32x32.mat')
            extra_x = extras['X'].swapaxes(0, 1).T
            extra_y = extras['y'].reshape((-1))

            train_x = np.concatenate((train_x, extra_x), axis=0)
            train_y = np.concatenate((train_y, extra_y), axis=0)

        if flatten is True:
            train_x = train_x.reshape((train_x.shape[0], -1))
            test_x = test_x.reshape((test_x.shape[0], -1))

        if gray is True:
            train_x = train_x[:, 0] * 0.2126 + \
                train_x[:, 1] * 0.7152 + train_x[:, 2] * 0.0722
            test_x = test_x[:, 0] * 0.2126 + \
                test_x[:, 1] * 0.7152 + test_x[:, 2] * 0.0722

            train_x = (train_x / 255.).reshape((len(train_x),
                                                32 * 32)).astype(np.float32)
            test_x = (test_x / 255.).reshape((len(test_x), 32 * 32)
                                             ).astype(np.float32)

        train_y[train_y == 10] = 0
        test_y[test_y == 10] = 0

        if toFloat:
            p.mean = np.mean(train_x, axis=0)
            p.std = np.sqrt(np.mean((train_x - p.mean[np.newaxis])**2, axis=0))
            train_x = ((train_x - p.mean[np.newaxis]) /
                       p.std[np.newaxis]).astype(np.float32)
            test_x = ((test_x - p.mean[np.newaxis]) /
                      p.std[np.newaxis]).astype(np.float32)
        if binarize_y:
            train_y = one_of_k(train_y)
            test_y = one_of_k(test_y)

        if test is True:
            #            model_path = "save.pkl"
            #            with open(model_path, "w") as f:
            #                pickle.dump([train_x, train_y, test_x, test_y, test_x, test_y], f)
            #            print "save save.pkl" %t

            return train_x, train_y, test_x, test_y, test_x, test_y

        else:
            return train_x, train_y

    def plot(X):
        if gray is True:
            X = X.reshape((X.shape[0], 32, 32))
            return X, "gray"
        else:
            X = X.reshape((X.shape[0], 3, 32, 32))
            if toFloat:
                X = (X * p.std[np.newaxis]) + p.mean[np.newaxis]
                X = X / 256.
                X[X < 0] = 0
                X[X > 1] = 1
            X = np.rollaxis(X, 1, 4)
            return X, None

    return load, plot

from sklearn.cross_validation import train_test_split


def lfw(datapath, toFloat=True, gray=False, rate=0.1, rseed=0):
    p = paramaters()

    def load(test=False):
        x = np.load(datapath + 'lfw_images.npy')
        y = np.load(datapath + 'lfw_attributes.npy').astype(np.float32)

        x = np.rollaxis(x, 3, 1)
        x = x[:, :, :61, :61]

        if gray:
            x = x[:, 0] * 0.2126 + x[:, 1] * 0.7152 + x[:, 2] * 0.0722
            x = x.reshape((len(x), 64 * 64)).astype(np.float32)

        if test:
            train_x, test_x, train_y, test_y = train_test_split(
                x, y, test_size=rate, random_state=rseed)
        else:
            train_x = x
            train_y = y

        if toFloat:
            p.mean = np.mean(train_x, axis=0)
            p.std = np.sqrt(np.mean((train_x - p.mean[np.newaxis])**2, axis=0))
            train_x = ((train_x - p.mean[np.newaxis]) /
                       p.std[np.newaxis]).astype(np.float32)
            if test:
                test_x = ((test_x - p.mean[np.newaxis]) /
                          p.std[np.newaxis]).astype(np.float32)
        else:
            train_x = train_x / 255.
            if test:
                test_x = test_x / 255.

        print train_x.shape

        if test:
            return train_x, train_y, test_x, test_y, test_x, test_y

        else:
            return train_x, train_y

    def preprocess(X):
        X = np.rollaxis(X, 3, 1)
        X = ((X - p.mean[np.newaxis]) / p.std[np.newaxis]).astype(np.float32)
        return X

    def plot(X):
        if gray is True:
            if toFloat:
                X = (X * p.std[np.newaxis]) + p.mean[np.newaxis]
                X = X / 255.
            X = X.reshape((X.shape[0], 61, 61))
            X[X < 0] = 0
            X[X > 1] = 1
            return X, "gray"
        else:
            X = X.reshape((X.shape[0], 3, 61, 61))
            if toFloat:
                X = (X * p.std[np.newaxis]) + p.mean[np.newaxis]
                X = X / 255.
            X = np.rollaxis(X, 1, 4)
            X[X < 0] = 0
            X[X > 1] = 1
            return X, None

    return load, plot, preprocess

def celeba(datapath, toFloat=True, gray=False, rate=0.001, rseed=0):
    p = paramaters()
    def load(test=False):
        x = np.load(datapath+'celeba_images.npy')
        y = np.load(datapath+'celeba_attributes.npy').astype(np.float32)

        x = np.rollaxis(x, 3, 1)
        x = x[:, :, :61, :61]

        if gray:
            x = x[:, 0] * 0.2126 + x[:, 1] * 0.7152 + x[:, 2] * 0.0722
            x = x.reshape((len(x), 64 * 64)).astype(np.float32)

        if test:
            train_x, test_x, train_y, test_y = train_test_split(
                x, y, test_size=rate, random_state=rseed)
        else:
            train_x = x
            train_y = y

        if toFloat:
            p.mean = np.mean(train_x, axis=0)
            p.std = np.sqrt(np.mean((train_x - p.mean[np.newaxis])**2, axis=0))
            train_x = ((train_x - p.mean[np.newaxis]) /
                       p.std[np.newaxis]).astype(np.float32)
            if test:
                test_x = ((test_x - p.mean[np.newaxis]) /
                          p.std[np.newaxis]).astype(np.float32)
        else:
            train_x = train_x / 255.
            if test:
                test_x = test_x / 255.

        print train_x.shape

        if test:
            return train_x, train_y, test_x, test_y, test_x, test_y

        else:
            return train_x, train_y

    def preprocess(X):
        X = np.rollaxis(X, 3, 1)
        X = ((X - p.mean[np.newaxis]) / p.std[np.newaxis]).astype(np.float32)
        return X

    def plot(X):
        if gray is True:
            if toFloat:
                X = (X * p.std[np.newaxis]) + p.mean[np.newaxis]
                X = X / 255.
            X = X.reshape((X.shape[0], 61, 61))
            X[X < 0] = 0
            X[X > 1] = 1
            return X, "gray"
        else:
            X = X.reshape((X.shape[0], 3, 61, 61))
            if toFloat:
                X = (X * p.std[np.newaxis]) + p.mean[np.newaxis]
                X = X / 255.
            X = np.rollaxis(X, 1, 4)
            X[X < 0] = 0
            X[X > 1] = 1
            return X, None

    return load, plot, preprocess
