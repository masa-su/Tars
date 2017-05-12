import gzip
import cPickle
import numpy as np
import scipy as sp
from scipy.io import loadmat
import glob
import sys
import pickle
from PIL import Image
from copy import copy

from sklearn.cross_validation import train_test_split

sys.setrecursionlimit(5000)


def one_of_k(a):
    a = np.array(a)
    b = np.zeros((a.size, 10)).astype('float32')
    b[np.arange(a.size), a] = 1
    return b


def mnist(datapath, toFloat=False):
    p = paramaters()

    def load(test=False, one_hot=True):
        f = gzip.open(datapath + 'mnist.pkl.gz', 'rb')
        (train_x, train_y), (valid_x, valid_y), (test_x, test_y) =\
            cPickle.load(f)
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


def lfw(datapath, toFloat=True, gray=False, rate=0.1, rseed=0):
    p = paramaters()

    def load(test=False):
        x = np.load(datapath + 'lfw_images.npy')
        y = np.load(datapath + 'lfw_attributes.npy').astype(np.float32)

        x = np.rollaxis(x, 3, 1)

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
            X = X.reshape((X.shape[0], 64, 64))
            X[X < 0] = 0
            X[X > 1] = 1
            return X, "gray"
        else:
            X = X.reshape((X.shape[0], 3, 64, 64))
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
        x = np.load(datapath + 'celeba_images.npy')
        y = np.load(datapath + 'celeba_attributes.npy').astype(np.float32)

        x = np.rollaxis(x, 3, 1)

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
            X = X.reshape((X.shape[0], 64, 64))
            X[X < 0] = 0
            X[X > 1] = 1
            return X, "gray"
        else:
            X = X.reshape((X.shape[0], 3, 64, 64))
            if toFloat:
                X = (X * p.std[np.newaxis]) + p.mean[np.newaxis]
                X = X / 255.
            X = np.rollaxis(X, 1, 4)
            X[X < 0] = 0
            X[X > 1] = 1
            return X, None

    return load, plot, preprocess


def flickr(datapath):
    def load(version=1, label=True, toFloat=True, raw_image=True):
        if label:
            train_indices = np.load(
                datapath + "flickr/splits/train_indices_%d.npy" % version)
            valid_indices = np.load(
                datapath + "flickr/splits/valid_indices_%d.npy" % version)
            test_indices = np.load(
                datapath + "flickr/splits/test_indices_%d.npy" % version)

            if raw_image:
                x_labelled = np.load(
                    datapath + "flickr/image/labelled/images.npy")

            else:
                x_labelled_path = glob.glob(datapath + "flickr/image/labelled/combined-*")
                x_labelled_path.sort()
                x_labelled = np.load(x_labelled_path[0])
                for path in x_labelled_path[1:]:
                    x_labelled = np.r_[x_labelled, np.load(path)]

            trn = []
            val = []
            tst = []

            y_ = np.load(datapath + "flickr/labels.npy").astype(np.float32)
            trn.append(y_[train_indices])
            val.append(y_[valid_indices])
            tst.append(y_[test_indices])

            trn.append(x_labelled[train_indices])
            val.append(x_labelled[valid_indices])
            tst.append(x_labelled[test_indices])

            w_labelled = LoadSparse(
                datapath + 'flickr/text/text_all_2000_labelled.npz')
            w_labelled = np.asarray(w_labelled.todense()).astype(np.float32)
            trn.append(w_labelled[train_indices])
            val.append(w_labelled[valid_indices])
            tst.append(w_labelled[test_indices])

            xw_labelled = np.c_[x_labelled, w_labelled]
            trn.append(xw_labelled[train_indices])
            val.append(xw_labelled[valid_indices])
            tst.append(xw_labelled[test_indices])

            if toFloat:
                mean = np.mean(trn[1], axis=0)
                std = np.sqrt(
                    np.mean((trn[1] - mean[np.newaxis, :])**2, axis=0))
                trn[1] = ((trn[1] - mean[np.newaxis, :]) /
                          std[np.newaxis, :]).astype(np.float32)
                val[1] = ((val[1] - mean[np.newaxis, :]) /
                          std[np.newaxis, :]).astype(np.float32)
                tst[1] = ((tst[1] - mean[np.newaxis, :]) /
                          std[np.newaxis, :]).astype(np.float32)

                """
                mean = np.mean(trn[3], axis=0)
                std = np.sqrt(
                    np.mean((trn[3] - mean[np.newaxis, :])**2, axis=0))
                trn[3] = ((trn[3] - mean[np.newaxis, :]) /
                          std[np.newaxis, :]).astype(np.float32)
                val[3] = ((val[3] - mean[np.newaxis, :]) /
                          std[np.newaxis, :]).astype(np.float32)
                tst[3] = ((tst[3] - mean[np.newaxis, :]) /
                          std[np.newaxis, :]).astype(np.float32)
                """
            return trn, val, tst

        else:
            model_path = datapath + \
                "flickr/image/unlabelled/unlabelled_trn.pkl"
            unlabel_trn = pickle.load(open(model_path))

            model_path = datapath + \
                "flickr/image/unlabelled/unlabelled_tst.pkl"
            unlabel_tst = pickle.load(open(model_path))

            return unlabel_trn, unlabel_tst

    def plot():
        pass

    def LoadSparse(inputfile, verbose=False):
        """Loads a sparse matrix stored as npz file."""
        npzfile = np.load(inputfile)
        mat = sp.sparse.csr_matrix(
            (npzfile['data'], npzfile['indices'], npzfile['indptr']),
            shape=tuple(list(npzfile['shape'])))
        if verbose:
            print 'Loaded sparse matrix from %s of shape %s' % (
                inputfile,
                mat.shape.__str__())
        return mat

    def shuffle(trn, each_permutation=False):
        trn = copy(trn)

        change_num = np.random.permutation(trn[0].shape[0])

        for i in range(len(trn)):
            if trn[i] is not None:
                trn[i] = trn[i][change_num]

        return trn

    return load, shuffle, plot


def facade(datapath):
    def load(label=True, test=True, crop=True):
        # Ref to https://github.com/pfnet-research/chainer-pix2pix
        # /blob/master/facade_dataset.py
        x = []
        y = []
        MAX_ITER = 378
        IMAGE_SHAPE = 256
        for i in range(1, MAX_ITER + 1):
            img = Image.open(datapath + "facade/base/cmp_b%04d.jpg" % i)
            label = Image.open(datapath + "facade/base/cmp_b%04d.png" % i)
            w, h = img.size
            r = 286. / min(w, h)
            img = img.resize((int(r * w), int(r * h)), Image.BILINEAR)
            label = label.resize((int(r * w), int(r * h)), Image.NEAREST)

            img = np.asarray(img).astype(
                "float32").transpose(2, 0, 1) / 128.0 - 1.0
            label_ = np.asarray(label) - 1
            label = np.zeros((12, img.shape[1], img.shape[2])).astype("int32")
            for j in range(12):
                label[j, :] = label_ == j

            # crop images
            img = img[:, :IMAGE_SHAPE, :IMAGE_SHAPE]
            label = label[:, :IMAGE_SHAPE, :IMAGE_SHAPE]
            x.append(img)
            y.append(label)

        x = np.asarray(x).astype("float32")
        y = np.asarray(y).astype("float32")

        train_x, train_y = x[:300], y[:300]
        test_x, test_y = x[300:], y[300:]

        if test:
            return train_x, train_y, test_x, test_y

        else:
            return train_x, train_y

    def plot(img):
        if img.shape[1] == 3:
            x = np.asarray(np.clip(img * 128 + 128, 0.0, 255.0),
                           dtype=np.uint8)
            x = x.transpose(0, 2, 3, 1)
            return x

        elif img.shape[1] == 12:
            x = np.ones((len(img), 3, 256, 256)).astype(np.uint8)
            for i in range(12):
                x[:, 0, :, :] += np.uint8(15 * i * img[:, i, :, :])
            x = x.transpose(0, 2, 3, 1)
            return x

        else:
            NotImplementedError

    return load, plot
