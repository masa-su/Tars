import theano.tensor as T

from ..utils import (
    gauss_unitgauss_kl,
    gauss_gauss_kl,
    epsilon
)
from .distribution_samples import (
    UnitGaussian_sample,
    UnitBernoulli_sample,
    UnitCategorical_sample,
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
            (1 - mean) * (T.log(1 - mean + epsilon()) + T.log(2))
        return T.sum(output, axis=1)

    elif q_class == "Categorical":
        mean = q.fprop(x, deterministic=deterministic)
        output = mean * (T.log(mean + epsilon()) + T.log(q.k))
        return T.sum(output, axis=1)

    elif q_class == "Kumaraswamy":
        # Kumaraswamy and beta
        prior_alpha = 1.
        prior_beta = 5.
        m = 10
        gamma = 0.57721

        a, b = q.fprop(x, deterministic=deterministic)

        def taylor(i, a, b):
            return 1. / (i + a * b) * beta(1 / a, b)
        kl, updates = theano.scan(fn=taylor,
                                  sequences=T.arange(m),
                                  non_sequences=[a, b])
        kl *= (prior_beta - 1) * b

        # Because T.psi haven't implemented yet.
        psi = T.log(b) - 1. / (2 * b) - 1. / (12 * b**2)
        kl += (a - prior_alpha) / a * (-gamma - psi - 1 / b)
        kl += T.log(a * b) + T.log(beta(prior_alpha, prior_beta))
        kl += -(b - 1) / b
        return T.sum(kl, axis=1)

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
        return UnitCategorical_sample(q.k)

    raise Exception("You cannot use this distribution as q")
