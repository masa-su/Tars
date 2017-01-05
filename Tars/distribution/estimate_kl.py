import math
import theano
import theano.tensor as T

from ..utils import epsilon
from .distribution_samples import (
    UnitGaussian_sample,
    UnitBernoulli_sample,
    UnitCategorical_sample,
    UnitBeta_sample,
    UnitDirichlet_sample,
    UnitGamma_sample,
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
        [Naelisnick+ 2016]
        Deep Generative Models with Stick-Breaking Priors
        """
        M = 10
        euler_gamma = 0.57721

        a, b = q1.fprop(x1, deterministic=deterministic)

        def taylor(m, a, b):
            return 1. / (m + a * b) * q2._beta_func(m / a, b)
        kl, _ = theano.scan(fn=taylor,
                            sequences=T.arange(1, M + 1),
                            non_sequences=[a, b])
        kl = T.sum(kl, axis=0)
        kl *= (q2.beta - 1) * b

        kl += ((a - q2.alpha) / a + epsilon()) *\
              (-euler_gamma - psi(b) - 1. / (b + epsilon()))
        kl += T.log(a * b + epsilon()) +\
            T.log(q2._beta_func(q2.alpha, q2.beta) + epsilon())
        kl += -(b - 1) / (b + epsilon())

        return T.sum(kl, axis=1)

    elif q1_class == "Gamma" and q2_class == "UnitGamma_sample":
        """
        https://arxiv.org/pdf/1611.01437.pdf
        """
        alpha1, beta1 = q1.fprop(x1, deterministic=deterministic)
        alpha2 = T.ones_like(alpha1)
        beta2 = T.ones_like(beta1)

        output = (alpha1 - alpha2) * psi(alpha1)
        output += -T.gammaln(alpha1) + T.gammaln(alpha2)
        output += alpha2 * (T.log(beta1 + epsilon()) -
                            T.log(beta2 + epsilon()))
        output += alpha1 * (beta2 - beta1) / (beta1 + epsilon())

        return T.sum(output, axis=1)

    elif q1_class == "Beta" and q2_class == "UnitGamma_sample":
        """
        http://bariskurt.com/kullback-leibler-divergence\
        -between-two-dirichlet-and-beta-distributions/
        """
        alpha1, beta1 = q1.fprop(x1, deterministic=deterministic)
        alpha2 = T.ones_like(alpha1) * q2_class.alpha
        beta2 = T.ones_like(beta1) * q2_class.beta

        output = T.gammaln(alpha1 + beta1) -\
            T.gammaln(alpha2 + beta2) -\
            (T.gammaln(alpha1) + T.gammaln(beta1)) +\
            (T.gammaln(alpha2) + T.gammaln(beta2)) +\
            (alpha1 - alpha2) * (psi(alpha1) - psi(alpha1 + beta1)) +\
            (beta1 - beta2) * (psi(beta1) - psi(alpha1 + beta1))

        return T.sum(output, axis=1)

    elif q1_class == "Dirichlet" and q2_class == "UnitDirichlet_sample":
        """
        http://bariskurt.com/kullback-leibler-divergence\
        -between-two-dirichlet-and-beta-distributions/
        """
        alpha1 = q1.fprop(x1, deterministic=deterministic)
        alpha2 = T.ones_like(alpha1) * q2_class.alpha

        output = T.gammaln(T.sum(alpha1, axis=-1)) -\
            T.gammaln(T.sum(alpha2, axis=-1)) -\
            T.sum(T.gammaln(alpha1), axis=-1) +\
            T.sum(T.gammaln(alpha2), axis=-1) +\
            T.sum((alpha1 - alpha2) *
                  (psi(alpha1) -
                   psi(T.sum(alpha1, axis=-1,
                             keepdims=True))), axis=-1)

        return T.sum(output, axis=1)

    raise Exception("You cannot use this distribution as q or prior, "
                    "got %s and %s." % (q1_class, q2_class))


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


def psi(b):
    # Because T.psi haven't implemented yet.
    output = T.log(b + epsilon()) - 1. / (2 * b + epsilon()) -\
        1. / (12 * b**2 + epsilon())
    return output


def get_prior(q):
    q_class = q.__class__.__name__
    if q_class == "Gaussian":
        return UnitGaussian_sample()

    elif q_class == "Bernoulli":
        return UnitBernoulli_sample()

    elif q_class == "Categorical":
        return UnitCategorical_sample(q.k)

    elif q_class == "Kumaraswamy" or q_class == "Beta":
        return UnitBeta_sample()

    elif q_class == "Dirichlet":
        return UnitDirichlet_sample()

    elif q_class == "Gamma":
        return UnitGamma_sample()

    raise Exception("You cannot use this distribution as q, "
                    "got %s." % q_class)
