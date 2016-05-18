import numpy as np
import theano 
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from util import gaussian_like
import lasagne

class Distribution(object):
    def __init__(self,mean_network,given):
        self.mean_network = mean_network
        self.given = given
        self.inputs = [x.input_var for x in given]

    def mean(self,x,deterministic=False):
        inputs = dict(zip(self.given,x))
        mean = lasagne.layers.get_output(self.mean_network,inputs,deterministic=deterministic)
        params = lasagne.layers.get_all_params(self.mean_network,trainable=True)
        return mean, params

    def mean_sum_samples(self,samples):
        n_dim = samples.ndim
        if n_dim == 2:
            return T.sum(samples, axis=1)
        elif n_dim == 4:
            return T.mean(T.sum(T.sum(samples, axis=2),axis=2),axis=1)

class Bernoulli(Distribution):
    def __init__(self,mean_network,given):
        super(Bernoulli, self).__init__(mean_network,given)
        
    def mean(self,x,deterministic=False):
        mean,params = super(Bernoulli, self).mean(x,deterministic)
        return mean, params

    def sample(self, mean, srng):
        return T.cast(T.le(srng.uniform(mean.shape), mean), mean.dtype)

    def log_likelihood(self, samples, mean):
        loglike = samples * T.log(mean) + (1 - samples) * T.log(1 - mean)
        return self.mean_sum_samples(loglike)

    def sample_given_x(self, x, srng, deterministic=False):
        mean, _ = self.mean(x,deterministic=deterministic)
        return self.sample(mean, srng)

    def sample_mean_given_x(self, x, deterministic=False):
        mean, _ = self.mean(x,deterministic=deterministic)
        return mean

    def log_likelihood_given_x(self, samples, x, deterministic=False):
        mean, params = self.mean(x,deterministic=deterministic)        
        return self.log_likelihood(samples, mean), params

class Categorical(Bernoulli):
    def __init__(self,mean_network,given):
        super(Categorical, self).__init__(mean_network,given)

    def log_likelihood(self, samples, mean):
        loglike = samples * T.log(mean)
        return self.mean_sum_samples(loglike)

class Gaussian(Distribution):
    def __init__(self,mean_network,var_network,given):
        super(Gaussian, self).__init__(mean_network,given)
        self.var_network = var_network
        
    def mean(self,x,deterministic=False):
        mean,params = super(Gaussian, self).mean(x,deterministic)
        inputs = dict(zip(self.given,x))
        var = lasagne.layers.get_output(self.var_network,inputs,deterministic=deterministic) # simga**2
        params += self.var_network.get_params(trainable=True)
        return mean, var, params

    def sample(self,mean, var, srng):
        eps = srng.normal(mean.shape)
        return mean + T.sqrt(var) * eps

    def log_likelihood(self, samples, mean, var):
        loglike = gaussian_like(samples, mean, var)
        return self.mean_sum_samples(loglike)

    def sample_given_x(self,x,srng,deterministic=False):
        mean, var, _ = self.mean(x, deterministic=deterministic)
        return self.sample(mean, var, srng)

    def sample_mean_given_x(self,x,deterministic=False):
        mean, _, _ = self.mean(x, deterministic=deterministic)
        return mean

    def log_likelihood_given_x(self, samples, x, deterministic=False):
        mean, var, params = self.mean(x, deterministic=deterministic)
        return self.log_likelihood(samples,mean,var), params

class Gaussian_nonvar(Bernoulli):
    def __init__(self,mean_network,given):
        super(Gaussian_nonvar, self).__init__(mean_network,given)

    def log_likelihood(self, samples, mean):
        loglike = gaussian_like(samples, mean, T.ones_like(mean))
        return self.mean_sum_samples(loglike)

class UnitGaussian(Distribution):
    def __init__(self):
        pass
        
    def sample(self, shape, srng):
        return srng.normal(shape)

    def log_likelihood(self, samples):
        loglike = gaussian_like(samples,T.zeros_like(samples),T.ones_like(samples))
        return T.mean(self.mean_sum_samples(loglike))
