import theano.tensor as T

from ..utils import (
    gauss_unitgauss_kl,
    gauss_gauss_kl,
    epsilon
)
from .distribution_samples import (
    UnitGaussian_sample,
    UnitBernoulli_sample,
)

def kl_vs_prior(q, x, deterministic=False):
    q_class = q.__class__.__name__
    if q_class == "Gaussian":
        mean, var = q.fprop(x, deterministic=deterministic)
        output = gauss_unitgauss_kl(mean, var)
        return T.sum(output, axis=1)

    elif q_class == "Bernoulli":
        mean = q.fprop(x, deterministic=deterministic)
        output = mean * (T.log(mean + epsilon()) + T.log(2)) +\
                 (1- mean) * (T.log(1 - mean + epsilon()) + T.log(2))
        return T.sum(output, axis=1)

    elif q_class == "Categorical":
        mean = q.fprop(x, deterministic=deterministic)
        output = mean * (T.log(mean + epsilon()) + T.log(q.k))
        return T.sum(output, axis=1)

    raise Exception("You cannot use this distribution as q")


def kl_vs_posterior(q1, q2, x1, x2, deterministic=False):
    q1_class = q1.__class__.__name__
    q2_class = q2.__class__.__name__
    if q1_class == "Gaussian" and q2_class == "Gaussian":
        mean1, var1 = q1.fprop(x1, deterministic=deterministic)
        mean2, var2 = q2.fprop(x2, deterministic=deterministic)
        return gauss_gauss_kl(mean1, var1, mean2, var2)

    raise Exception("You cannot use these distributions as q")

def set_prior(q):
    q_class = q.__class__.__name__
    if q_class == "Gaussian":
        return UnitGaussian_sample()

    elif q_class == "Bernoulli":
        return UnitBernoulli_sample()

    elif q_class == "Categorical":
        return UnitGaussian_sample()

    raise Exception("You cannot use this distribution as q")
