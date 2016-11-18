import math
import theano
import theano.tensor as T

from ..utils import epsilon
from .distribution_samples import (
    UnitGaussian_sample,
    UnitBernoulli_sample,
    UnitCategorical_sample,
    UnitBeta_sample,
)


def analytical_kl(q1, q2, given, deterministic=False):
    try:
        [x1, x2] = given
    except:
        raise ValueError("The length of given list must be 2, "
                         "got %d" % len(given))

    q1_class = q1.__class__.__name__
    q2_class = q2.__class__.__name__
    if q1_class == "Gaussian" and q2_class == "UnitGaussian_sample":
        mean, var = q1.fprop(x1, deterministic=deterministic)
        return gauss_unitgauss_kl(mean, var)

    if q1_class == "Gaussian" and q2_class == "Gaussian":
        mean1, var1 = q1.fprop(x1, deterministic=deterministic)
        mean2, var2 = q2.fprop(x2, deterministic=deterministic)
        return gauss_gauss_kl(mean1, var1, mean2, var2)

    elif q1_class == "Bernoulli" and q2_class == "UnitBernoulli_sample":
        mean = q1.fprop(x1, deterministic=deterministic)
        output = mean * (T.log(mean + epsilon()) + T.log(2)) +\
            (1 - mean) * (T.log(1 - mean + epsilon()) + T.log(2))
        return T.sum(output, axis=1)

    elif q1_class == "Categorical" and q2_class == "UnitCategorical_sample":
        mean = q1.fprop(x1, deterministic=deterministic)
        output = mean * (T.log(mean + epsilon()) + T.log(q1.k))
        return T.sum(output, axis=1)

    elif q1_class == "Kumaraswamy" and q2_class == "UnitBeta_sample":
        """
        [Naelisnick+ 2016] Deep Generative Models with Stick-Breaking Priors
        """
        m = 10
        gamma = 0.57721

        a, b = q1.fprop(x1, deterministic=deterministic)

        def taylor(i, a, b):
            return 1. / (i + a * b) * q2._beta_func(1 / a, b)
        kl, updates = theano.scan(fn=taylor,
                                  sequences=T.arange(m),
                                  non_sequences=[a, b])
        kl *= (q2.beta - 1) * b

        # Because T.psi haven't implemented yet.
        psi = T.log(b) - 1. / (2 * b) - 1. / (12 * b**2)
        kl += (a - q2.alpha) / a * (-gamma - psi - 1 / b)
        kl += T.log(a * b) + T.log(q2._beta_func(q2.alpha, q2.beta))
        kl += -(b - 1) / b
        return T.sum(kl, axis=1)

    raise Exception("You cannot use this distribution as q")


def gaussian_like(x, mean, var):
    c = - 0.5 * math.log(2 * math.pi)
    _var = var + epsilon()  # avoid NaN
    return c - T.log(_var) / 2 - (x - mean)**2 / (2 * _var)


def gauss_unitgauss_kl(mean, var):
    return -0.5 * T.sum(1 + T.log(var) - mean**2 - var, axis=1)


def gauss_gauss_kl(mean1, var1, mean2, var2):
    _var2 = var2 + epsilon()  # avoid NaN
    _kl = T.log(var2) - T.log(var1) \
        + (var1 + (mean1 - mean2)**2) / _var2 - 1
    return 0.5 * T.sum(_kl, axis=1)


def set_prior(q):
    q_class = q.__class__.__name__
    if q_class == "Gaussian":
        return UnitGaussian_sample()

    elif q_class == "Bernoulli":
        return UnitBernoulli_sample()

    elif q_class == "Categorical":
        return UnitCategorical_sample(q.k)

    elif q_class == "UnitBeta_sample":
        return UnitBeta_sample()

    raise Exception("You cannot use this distribution as q")
